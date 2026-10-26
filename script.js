/* ============================================================
   MOTOR DE SCORING CREDITICIO
   Datos, métricas y conexión con FastAPI
============================================================ */

const API_BASE = 'http://127.0.0.1:8000';
const PAGE_SIZE = 25;

const appState = {
  dataRows: [],
  filteredRows: [],
  dataColumns: [],
  currentFile: 'clientes_solicitud_credito.csv',
  currentPage: 1,

  // Escenario seleccionado en la interfaz.
  currentScenario: '5',

  // Información del modelo que realmente está cargado en FastAPI.
  modelInfo: null,

  // Filas del CSV del escenario actualmente seleccionado.
  modelRows: [],

  charts: {
    discrimination: null,
    classification: null
  }
};

const DATA_DESCRIPTIONS = {
  raw: 'Dataset original con valores e inconsistencias introducidos para probar la limpieza.',
  clean: 'Dataset resultante del proceso de limpieza e imputación.',
  woe: 'Diccionario de intervalos, frecuencias y valores WoE del modelo exportado.'
};

const METRIC_COLUMNS = [
  ['AUC_media', 'AUC-ROC'],
  ['Gini_media', 'Gini'],
  ['KS_media', 'KS'],
  ['Accuracy_media', 'Accuracy'],
  ['Precision_media', 'Precision'],
  ['Recall_media', 'Recall'],
  ['F1_media', 'F1-score'],
  ['Brier_media', 'Brier']
];


/* ============================================================
   1. UTILIDADES
============================================================ */

function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[character]));
}

function cleanHeader(value) {
  return String(value ?? '')
    .replace(/^\uFEFF/, '')
    .replace(/\*/g, '')
    .trim();
}

function normalizeRow(row) {
  const result = {};

  Object.entries(row || {}).forEach(([key, value]) => {
    result[cleanHeader(key)] = value;
  });

  return result;
}

function numericValue(value) {
  if (value === null || value === undefined || value === '') {
    return NaN;
  }

  const number = Number(value);

  return Number.isFinite(number) ? number : NaN;
}

function formatMetric(value, digits = 4) {
  const number = numericValue(value);

  return Number.isFinite(number)
    ? number.toFixed(digits)
    : '—';
}


/* ============================================================
   2. IDENTIFICACIÓN DEL MODELO Y ESCENARIO
============================================================ */

// Quita acentos y caracteres para comparar nombres de modelos.
function normalizeModelName(name) {
  const normalized = String(name ?? '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

  // Compatibilidad con nombres habituales de XGBoost.
  if (
    normalized.includes('xgboost') ||
    normalized.includes('xgbclassifier')
  ) {
    return 'xgboost';
  }

  if (
    normalized.includes('randomforest') ||
    normalized.includes('randomforestclassifier')
  ) {
    return 'randomforest';
  }

  return normalized;
}

function modelNamesMatch(nameA, nameB) {
  return normalizeModelName(nameA) === normalizeModelName(nameB);
}


// Obtiene el escenario a partir de la metadata de FastAPI.
function inferScenarioFromModelInfo(info) {
  if (!info) return null;

  const scenarioText = String(info.scenario || '');

  // Ejemplos: "8_variables", "5_variables".
  const match = scenarioText.match(/(?:^|\D)(5|8)(?:\D|$)/);

  if (match) {
    return match[1];
  }

  // Si el nombre del escenario no permite identificarlo,
  // se utiliza el número real de variables del artefacto.
  const count = Number(
    info.feature_count ||
    (Array.isArray(info.variables) ? info.variables.length : 0)
  );

  if (count === 5 || count === 8) {
    return String(count);
  }

  return null;
}


function updateScenarioButtons(scenario) {
  document.querySelectorAll('.scenario-btn').forEach(button => {
    button.classList.toggle(
      'active',
      button.dataset.scenario === String(scenario)
    );
  });
}


/* ============================================================
   3. ESTADO DE CONEXIÓN CON FASTAPI
============================================================ */

function setStatus(message, state = 'pending', detail = '') {
  const label = document.getElementById('api-status');
  const dot = document.getElementById('api-dot');
  const detailLabel = document.getElementById('api-status-detail');

  if (label) {
    label.textContent = message;
  }

  if (detailLabel) {
    detailLabel.textContent = detail || API_BASE;
  }

  if (dot) {
    dot.classList.remove('online', 'offline');

    if (state === 'online') {
      dot.classList.add('online');
    }

    if (state === 'offline') {
      dot.classList.add('offline');
    }
  }
}


// Comprueba /health y después consulta /model-info.
async function checkApi() {
  try {
    const response = await fetch(
      `${API_BASE}/health`,
      { cache: 'no-store' }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const health = await response.json();

    if (health.model_loaded) {
      setStatus(
        'API conectada',
        'online',
        health.model_name || 'Modelo cargado'
      );
    } else {
      setStatus(
        'API activa · falta modelo',
        'offline',
        health.error || 'Revisa el artefacto .joblib'
      );
    }

  } catch (error) {
    setStatus(
      'API desconectada',
      'offline',
      'Ejecuta python api.py'
    );

    console.warn('No se pudo consultar /health:', error);
  }

  const info = await loadModelInfo();

  const scenario = inferScenarioFromModelInfo(info);

  if (info && info.model_loaded && scenario) {
    // El escenario se determina a partir del modelo real,
    // no de un valor predeterminado.
    appState.currentScenario = scenario;

    updateScenarioButtons(scenario);

    loadModelMetrics(scenario);

  } else {
    // Permite consultar los CSV aunque la API esté apagada,
    // pero no atribuye métricas a un modelo no identificado.
    loadModelMetrics(appState.currentScenario);
    renderActiveModelMetrics(null);
  }
}


// Obtiene nombre, escenario, variables y transformación.
async function loadModelInfo() {
  try {
    const response = await fetch(
      `${API_BASE}/model-info`,
      { cache: 'no-store' }
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const info = await response.json();

    appState.modelInfo = info;

    document.getElementById('loaded-model-tag').textContent =
      info.model_loaded ? 'Cargado' : 'No disponible';

    document.getElementById('loaded-model-name').textContent =
      info.model_name || '—';

    document.getElementById('loaded-model-scenario').textContent =
      info.scenario || '—';

    document.getElementById('loaded-model-features').textContent =
      Array.isArray(info.variables)
        ? `${info.feature_count ?? info.variables.length} · ${info.variables.join(', ')}`
        : '—';

    document.getElementById('loaded-model-woe').textContent =
      info.uses_woe
        ? 'Binning + WoE'
        : 'Variables originales';

    return info;

  } catch (error) {
    appState.modelInfo = null;

    document.getElementById('loaded-model-tag').textContent =
      'API no disponible';

    document.getElementById('loaded-model-name').textContent =
      'No se pudo consultar';

    document.getElementById('loaded-model-scenario').textContent = '—';
    document.getElementById('loaded-model-features').textContent = '—';
    document.getElementById('loaded-model-woe').textContent = '—';

    console.warn('Error consultando /model-info:', error);

    return null;
  }
}


/* ============================================================
   4. NAVEGACIÓN ENTRE PESTAÑAS
============================================================ */

function openTab(tabId, button) {
  document.querySelectorAll('.tab-content').forEach(section => {
    section.classList.remove('active');
  });

  document.querySelectorAll('.main-nav .tab-btn').forEach(tab => {
    tab.classList.remove('active');
  });

  const target = document.getElementById(tabId);

  if (target) {
    target.classList.add('active');
  }

  if (button) {
    button.classList.add('active');
  }

  // Recalcular el tamaño de los gráficos al hacer visible
  // la pestaña Modelos.
  if (tabId === 'tab-modelos') {
    Object.values(appState.charts).forEach(chart => {
      if (chart) chart.resize();
    });
  }
}


/* ============================================================
   5. EXPLORADOR DE DATASETS CSV
============================================================ */

function loadDataset(filename, button = null) {
  const container = document.getElementById('data-table-container');

  if (!container) return;

  if (typeof Papa === 'undefined') {
    container.innerHTML =
      '<div class="empty-state">No se pudo cargar PapaParse. Comprueba la librería CDN.</div>';
    return;
  }

  appState.currentFile = filename;
  appState.currentPage = 1;
  appState.dataRows = [];
  appState.filteredRows = [];
  appState.dataColumns = [];

  container.innerHTML =
    '<div class="empty-state">Leyendo el archivo...</div>';

  document.querySelectorAll('.source-btn').forEach(item => {
    item.classList.remove('active');
  });

  if (button) {
    button.classList.add('active');
  } else {
    const matching = [...document.querySelectorAll('.source-btn')]
      .find(item => item.dataset.file === filename);

    if (matching) {
      matching.classList.add('active');
    }
  }

  const kind = button?.dataset.kind ||
    (filename.includes('clean') ? 'clean' :
      filename.includes('woe') ? 'woe' : 'raw');

  document.getElementById('dataset-description').textContent =
    `${filename} · ${DATA_DESCRIPTIONS[kind] || ''}`;

  Papa.parse(filename, {
    download: true,
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,

    complete: results => {
      const rows = (results.data || [])
        .map(normalizeRow)
        .filter(row =>
          Object.values(row).some(
            value => String(value ?? '').trim() !== ''
          )
        );

      if (!rows.length) {
        container.innerHTML = `
          <div class="empty-state">
            No se encontraron registros en ${escapeHTML(filename)}.
            Comprueba que el archivo exista junto a index.html
            y que estés usando un servidor local.
          </div>
        `;

        resetDatasetStats();
        return;
      }

      appState.dataRows = rows;
      appState.dataColumns = Object.keys(rows[0]);
      appState.filteredRows = rows.slice();

      updateDatasetStats(rows, appState.dataColumns);
      renderDataTable();
    },

    error: error => {
      container.innerHTML = `
        <div class="empty-state">
          No se pudo leer ${escapeHTML(filename)}.
          Error: ${escapeHTML(error.message)}
        </div>
      `;

      resetDatasetStats();
    }
  });
}


function resetDatasetStats() {
  [
    'stat-rows',
    'stat-columns',
    'stat-nulls',
    'stat-default'
  ].forEach(id => {
    const element = document.getElementById(id);

    if (element) {
      element.textContent = '—';
    }
  });

  document.getElementById('table-count').textContent =
    'Archivo no disponible';

  document.getElementById('page-info').textContent = '—';
  document.getElementById('prev-page').disabled = true;
  document.getElementById('next-page').disabled = true;
}


function updateDatasetStats(rows, columns) {
  const rowCount = rows.length;
  let missing = 0;

  rows.forEach(row => {
    columns.forEach(column => {
      const value = row[column];

      if (
        value === null ||
        value === undefined ||
        String(value).trim() === ''
      ) {
        missing += 1;
      }
    });
  });

  let defaultRate = null;

  if (columns.includes('estado_default')) {
    const validTargets = rows
      .map(row => numericValue(row.estado_default))
      .filter(value => value === 0 || value === 1);

    if (validTargets.length > 0) {
      defaultRate =
        validTargets.reduce((sum, value) => sum + value, 0) /
        validTargets.length;
    }
  }

  document.getElementById('stat-rows').textContent =
    rowCount.toLocaleString('es-EC');

  document.getElementById('stat-columns').textContent =
    columns.length.toLocaleString('es-EC');

  document.getElementById('stat-nulls').textContent =
    missing.toLocaleString('es-EC');

  document.getElementById('stat-default').textContent =
    defaultRate === null
      ? 'N/D'
      : `${(defaultRate * 100).toFixed(2)}%`;
}


function renderDataTable() {
  const container = document.getElementById('data-table-container');

  if (!container || !appState.dataRows.length) return;

  const totalPages = Math.max(
    1,
    Math.ceil(appState.filteredRows.length / PAGE_SIZE)
  );

  appState.currentPage = Math.min(
    Math.max(1, appState.currentPage),
    totalPages
  );

  const start = (appState.currentPage - 1) * PAGE_SIZE;

  const visibleRows = appState.filteredRows.slice(
    start,
    start + PAGE_SIZE
  );

  if (!visibleRows.length) {
    container.innerHTML =
      '<div class="empty-state">No hay registros que coincidan con la búsqueda.</div>';
  } else {
    let html = '<table><thead><tr>';

    appState.dataColumns.forEach(column => {
      html += `<th>${escapeHTML(column)}</th>`;
    });

    html += '</tr></thead><tbody>';

    visibleRows.forEach(row => {
      html += '<tr>';

      appState.dataColumns.forEach(column => {
        const raw = row[column] ?? '';
        const number = numericValue(raw);

        const display =
          Number.isFinite(number) && String(raw).length > 8
            ? number.toFixed(4)
            : raw;

        html += `<td>${escapeHTML(display)}</td>`;
      });

      html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  }

  const from = appState.filteredRows.length > 0
    ? start + 1
    : 0;

  const to = Math.min(
    start + PAGE_SIZE,
    appState.filteredRows.length
  );

  document.getElementById('page-info').textContent =
    `${from}–${to} de ${appState.filteredRows.length.toLocaleString('es-EC')} filas filtradas`;

  document.getElementById('table-count').textContent =
    `${appState.dataRows.length.toLocaleString('es-EC')} registros cargados`;

  document.getElementById('prev-page').disabled =
    appState.currentPage <= 1;

  document.getElementById('next-page').disabled =
    appState.currentPage >= totalPages;
}


function filterDataTable(query) {
  const normalized = String(query ?? '').trim().toLowerCase();

  appState.filteredRows = !normalized
    ? appState.dataRows.slice()
    : appState.dataRows.filter(row =>
        appState.dataColumns.some(column =>
          String(row[column] ?? '').toLowerCase().includes(normalized)
        )
      );

  appState.currentPage = 1;
  renderDataTable();
}


/* ============================================================
   6. CARGA DE MÉTRICAS SEGÚN EL ESCENARIO
============================================================ */

function loadModelMetrics(scenario = appState.currentScenario) {
  appState.currentScenario = String(scenario);

  updateScenarioButtons(appState.currentScenario);

  const filename =
    `comparacion_modelos_${appState.currentScenario}_variables.csv`;

  const message = document.getElementById('model-load-message');
  const tableContainer = document.getElementById('model-table-container');

  tableContainer.innerHTML =
    '<div class="empty-state">Cargando métricas de validación cruzada...</div>';

  message.textContent =
    `Fuente: ${filename}. Los valores corresponden a promedios de validación cruzada.`;

  if (typeof Papa === 'undefined') {
    renderMissingMetrics(filename);
    return;
  }

  Papa.parse(filename, {
    download: true,
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,

    complete: results => {
      // Los nombres de columnas se normalizan para admitir
      // encabezados como **Brier_media** si contienen asteriscos.
      const rows = (results.data || [])
        .map(normalizeRow)
        .filter(row => row.Modelo);

      if (!rows.length) {
        appState.modelRows = [];
        renderMissingMetrics(filename);
        return;
      }

      appState.modelRows = rows.sort(
        (a, b) =>
          numericValue(b.AUC_media) -
          numericValue(a.AUC_media)
      );

      renderModelMetrics();
    },

    error: error => {
      appState.modelRows = [];
      renderMissingMetrics(filename, error.message);
    }
  });
}


function renderMissingMetrics(filename, reason = '') {
  const container = document.getElementById('model-table-container');

  container.innerHTML = `
    <div class="empty-state">
      No se pudo cargar ${escapeHTML(filename)}.
      Comprueba que esté en la carpeta del frontend.
      ${reason ? `<br>${escapeHTML(reason)}` : ''}
    </div>
  `;

  document.getElementById('model-load-message').textContent =
    'No hay datos disponibles para este escenario. No se inventan ni se sustituyen métricas.';

  renderActiveModelMetrics(null);

  destroyChart('discrimination');
  destroyChart('classification');
}


/* ============================================================
   7. TARJETAS DEL MODELO REALMENTE CARGADO
============================================================ */

function renderActiveModelMetrics(row) {
  document.getElementById('best-auc').textContent =
    row ? formatMetric(row.AUC_media) : '—';

  document.getElementById('best-gini').textContent =
    row ? formatMetric(row.Gini_media) : '—';

  document.getElementById('best-ks').textContent =
    row ? formatMetric(row.KS_media, 2) : '—';

  document.getElementById('best-brier').textContent =
    row ? formatMetric(row.Brier_media) : '—';

  const label = document.getElementById('best-auc-model');

  if (!row) {
    if (!appState.modelInfo || !appState.modelInfo.model_loaded) {
      label.textContent = 'API sin modelo activo';
    } else {
      label.textContent =
        'No hay métricas coincidentes para el modelo activo';
    }

    return;
  }

  label.textContent =
    `${appState.modelInfo.model_name} · Promedio CV`;
}


/* ============================================================
   8. TABLA COMPARATIVA DE MODELOS
============================================================ */

function renderModelMetrics() {
  const rows = appState.modelRows;

  if (!rows.length) return;

  const apiScenario = inferScenarioFromModelInfo(
    appState.modelInfo
  );

  // Las tarjetas solo pueden mostrar métricas del mismo escenario
  // y del mismo algoritmo que FastAPI tiene realmente cargados.
  const sameScenario =
    Boolean(appState.modelInfo?.model_loaded) &&
    apiScenario !== null &&
    apiScenario === appState.currentScenario;

  let activeModelRow = null;

  if (sameScenario) {
    activeModelRow = rows.find(row =>
      modelNamesMatch(
        row.Modelo,
        appState.modelInfo.model_name
      )
    ) || null;
  }

  renderActiveModelMetrics(activeModelRow);

  const container = document.getElementById('model-table-container');

  let html = `
    <table>
      <thead>
        <tr>
          <th>Modelo</th>
          <th>AUC-ROC</th>
          <th>Gini</th>
          <th>KS</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1-score</th>
          <th>Brier</th>
        </tr>
      </thead>
      <tbody>
  `;

  rows.forEach(row => {
    const isCurrent =
      sameScenario &&
      modelNamesMatch(
        row.Modelo,
        appState.modelInfo.model_name
      );

    html += `
      <tr${isCurrent ? ' class="current-model-row"' : ''}>
        <td>
          <strong>${escapeHTML(row.Modelo)}</strong>
          ${isCurrent ? ' <span class="mini-tag">API ACTIVA</span>' : ''}
        </td>
    `;

    METRIC_COLUMNS.forEach(([key]) => {
      const number = numericValue(row[key]);

      const digits = key === 'KS_media' ? 2 : 4;

      html += `<td>${
        Number.isFinite(number)
          ? number.toFixed(digits)
          : '—'
      }</td>`;
    });

    html += '</tr>';
  });

  html += '</tbody></table>';

  container.innerHTML = html;

  document.getElementById('cv-source-label').textContent =
    `Validación cruzada · ${appState.currentScenario} variables`;

  const message = document.getElementById('model-load-message');

  if (activeModelRow) {
    message.textContent =
      `Mostrando los resultados de validación cruzada de ${appState.modelInfo.model_name} en el escenario ${appState.currentScenario}. Su fila está marcada como API ACTIVA.`;

  } else if (appState.modelInfo?.model_loaded) {
    message.textContent =
      `La tabla muestra el escenario ${appState.currentScenario} para comparar modelos. El artefacto activo de FastAPI corresponde a ${appState.modelInfo.scenario}; por ello, las tarjetas del modelo activo no muestran métricas de este otro escenario.`;

  } else {
    message.textContent =
      'Se muestran las métricas del escenario seleccionado. La API no ha confirmado un modelo activo.';
  }

  renderMetricCharts(rows);
}


/* ============================================================
   9. GRÁFICOS
============================================================ */

function destroyChart(key) {
  if (appState.charts[key]) {
    appState.charts[key].destroy();
    appState.charts[key] = null;
  }
}


function renderMetricCharts(rows) {
  if (typeof Chart === 'undefined') return;

  destroyChart('discrimination');
  destroyChart('classification');

  const labels = rows.map(row => row.Modelo);

  const discriminationCanvas =
    document.getElementById('chart-discrimination');

  const classificationCanvas =
    document.getElementById('chart-classification');

  if (discriminationCanvas) {
    appState.charts.discrimination = new Chart(
      discriminationCanvas,
      {
        type: 'bar',

        data: {
          labels,

          datasets: [
            {
              label: 'AUC-ROC',
              data: rows.map(row => numericValue(row.AUC_media)),
              backgroundColor: '#2563eb',
              borderRadius: 5
            },
            {
              label: 'Gini',
              data: rows.map(row => numericValue(row.Gini_media)),
              backgroundColor: '#06b6d4',
              borderRadius: 5
            }
          ]
        },

        options: {
          responsive: true,
          maintainAspectRatio: false,

          scales: {
            y: {
              min: 0,
              max: 1,

              ticks: {
                callback: value => Number(value).toFixed(1)
              },

              grid: {
                color: '#edf2f7'
              }
            },

            x: {
              grid: {
                display: false
              },

              ticks: {
                font: {
                  size: 10
                }
              }
            }
          },

          plugins: {
            legend: {
              position: 'bottom',

              labels: {
                boxWidth: 10,
                usePointStyle: true
              }
            }
          }
        }
      }
    );
  }


  if (classificationCanvas) {
    appState.charts.classification = new Chart(
      classificationCanvas,
      {
        type: 'bar',

        data: {
          labels,

          datasets: [
            {
              label: 'Precision',
              data: rows.map(row => numericValue(row.Precision_media)),
              backgroundColor: '#8b5cf6',
              borderRadius: 5
            },
            {
              label: 'Recall',
              data: rows.map(row => numericValue(row.Recall_media)),
              backgroundColor: '#f59e0b',
              borderRadius: 5
            },
            {
              label: 'F1-score',
              data: rows.map(row => numericValue(row.F1_media)),
              backgroundColor: '#10b981',
              borderRadius: 5
            }
          ]
        },

        options: {
          responsive: true,
          maintainAspectRatio: false,

          scales: {
            y: {
              min: 0,
              max: 1,

              ticks: {
                callback: value => Number(value).toFixed(1)
              },

              grid: {
                color: '#edf2f7'
              }
            },

            x: {
              grid: {
                display: false
              },

              ticks: {
                font: {
                  size: 10
                }
              }
            }
          },

          plugins: {
            legend: {
              position: 'bottom',

              labels: {
                boxWidth: 10,
                usePointStyle: true
              }
            }
          }
        }
      }
    );
  }
}


/* ============================================================
   10. PREDICCIÓN CON FASTAPI
============================================================ */

async function submitPrediction(event) {
  event.preventDefault();

  const form = document.getElementById('prediction-form');
  const feedback = document.getElementById('form-feedback');
  const button = document.getElementById('predict-button');
  const panel = document.getElementById('result-panel');

  if (!form.reportValidity()) {
    return;
  }

  // El backend selecciona solo las variables que necesita
  // el artefacto activo. Se envían los campos disponibles.
  const payload = {
    edad_titular: Number(form.elements.edad_titular.value),

    ingreso_mensual_usd: Number(
      form.elements.ingreso_mensual_usd.value
    ),

    ratio_deuda_ingreso: Number(
      form.elements.ratio_deuda_ingreso.value
    ),

    antiguedad_laboral_meses: Number(
      form.elements.antiguedad_laboral_meses.value
    ),

    consultas_buro_ultimos_6m: Number(
      form.elements.consultas_buro_ultimos_6m.value
    ),

    meses_maximo_mora_historica: Number(
      form.elements.meses_maximo_mora_historica.value
    ),

    score_buro_externo: Number(
      form.elements.score_buro_externo.value
    ),

    monto_credito_solicitado: Number(
      form.elements.monto_credito_solicitado.value
    ),

    tipo_vivienda: form.elements.tipo_vivienda.value
  };


  // Validación lógica de antigüedad laboral.
  const maxAntiguedad = Math.max(
    0,
    (payload.edad_titular - 18) * 12
  );

  if (payload.antiguedad_laboral_meses > maxAntiguedad) {
    feedback.textContent =
      `La antigüedad laboral supera el máximo lógico de ${maxAntiguedad} meses según la edad.`;

    feedback.style.color = '#b91c1c';

    return;
  }


  button.disabled = true;
  button.querySelector('span:first-child').textContent =
    'Calculando...';

  feedback.textContent =
    'Enviando la solicitud al modelo Python...';

  feedback.style.color = '';

  panel.classList.remove('empty-result');

  panel.innerHTML =
    '<div class="empty-state">El modelo está calculando la probabilidad de default...</div>';


  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',

      headers: {
        'Content-Type': 'application/json'
      },

      body: JSON.stringify(payload)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(item => item.msg || JSON.stringify(item)).join('; ')
        : (data.detail || `Error HTTP ${response.status}`);

      throw new Error(detail);
    }

    renderPredictionResult(data);

    feedback.textContent =
      'Predicción recibida correctamente desde FastAPI.';

    feedback.style.color = '#15803d';

  } catch (error) {
    panel.classList.add('empty-result');

    panel.innerHTML = `
      <div class="result-placeholder">
        <span class="result-symbol">!</span>

        <h3>No se pudo calcular</h3>

        <p>
          ${escapeHTML(error.message)}
          Comprueba que el servidor esté activo y el artefacto cargado.
        </p>
      </div>
    `;

    feedback.textContent =
      'La solicitud no se completó.';

    feedback.style.color = '#b91c1c';

  } finally {
    button.disabled = false;

    button.querySelector('span:first-child').textContent =
      'Calcular probabilidad';
  }
}


/* ============================================================
   11. MOSTRAR RESULTADO DE LA PREDICCIÓN
============================================================ */

function renderPredictionResult(data) {
  const panel = document.getElementById('result-panel');

  const pd = numericValue(data.probabilidad_default);

  if (!Number.isFinite(pd)) {
    throw new Error('La API no devolvió una probabilidad válida.');
  }

  const banda = String(data.banda || 'Riesgo no clasificado');

  const bandMatch = banda.match(/^([A-D])/i);

  const bandLetter = bandMatch
    ? bandMatch[1].toLowerCase()
    : 'd';

  const meterWidth = Math.max(0, Math.min(100, pd));

  const threshold = numericValue(data.umbral_modelo);

  const thresholdText = Number.isFinite(threshold)
    ? `${threshold.toFixed(2)} %`
    : 'N/D';

  const variables = Array.isArray(data.variables_utilizadas)
    ? data.variables_utilizadas.join(', ')
    : 'N/D';

  panel.classList.remove('empty-result');

  panel.innerHTML = `
    <div style="width:100%">

      <div class="result-header">

        <div>
          <p class="eyebrow">RESULTADO DE INFERENCIA</p>

          <h3>
            ${escapeHTML(data.decision || 'Resultado calculado')}
          </h3>
        </div>

        <span class="risk-badge risk-${escapeHTML(bandLetter)}">
          ${escapeHTML(banda.split(' - ')[0])}
        </span>

      </div>


      <div class="pd-value">
        ${pd.toFixed(2)}%
      </div>

      <div class="pd-caption">
        Probabilidad estimada de default a 12 meses (PD)
      </div>


      <div
        class="pd-meter"
        role="progressbar"
        aria-label="Probabilidad de default"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow="${meterWidth}">

        <span style="width:${meterWidth}%"></span>

      </div>


      <div class="result-detail">
        <span>Banda de riesgo</span>
        <strong>${escapeHTML(banda)}</strong>
      </div>


      <div class="result-divider"></div>


      <div class="result-details">

        <div class="result-detail">
          <span>Modelo activo</span>
          <strong>${escapeHTML(data.modelo || 'N/D')}</strong>
        </div>

        <div class="result-detail">
          <span>Escenario</span>
          <strong>${escapeHTML(data.escenario || 'N/D')}</strong>
        </div>

        <div class="result-detail">
          <span>Umbral binario</span>
          <strong>${thresholdText}</strong>
        </div>

        <div class="result-detail">
          <span>Clasificación por umbral</span>
          <strong>${escapeHTML(data.clasificacion_umbral || 'N/D')}</strong>
        </div>

        <div class="result-detail">
          <span>Transformación</span>
          <strong>${data.usa_woe ? 'Binning + WoE' : 'Variables originales'}</strong>
        </div>

      </div>


      <div class="result-divider"></div>


      <p class="footnote">
        <strong>Variables utilizadas:</strong>
        ${escapeHTML(variables)}
      </p>

      <p class="footnote">
        ${escapeHTML(
          data.nota_politica ||
          'Resultado demostrativo; no equivale a una aprobación crediticia real.'
        )}
      </p>

    </div>
  `;
}


function resetPrediction() {
  const feedback = document.getElementById('form-feedback');
  const panel = document.getElementById('result-panel');

  feedback.textContent = '';
  feedback.style.color = '';

  panel.classList.add('empty-result');

  panel.innerHTML = `
    <div class="result-placeholder">

      <span class="result-symbol">%</span>

      <h3>Esperando una solicitud</h3>

      <p>
        Completa los campos y ejecuta el modelo para consultar
        la PD estimada a 12 meses.
      </p>

    </div>
  `;
}


/* ============================================================
   12. INICIALIZACIÓN
============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // Navegación principal.
  document.querySelectorAll('.main-nav .tab-btn').forEach(button => {
    button.addEventListener('click', () => {
      openTab(button.dataset.tab, button);
    });
  });


  // Selector de archivos CSV.
  document.querySelectorAll('.source-btn').forEach(button => {
    button.addEventListener('click', () => {
      loadDataset(button.dataset.file, button);
    });
  });


  // Búsqueda de registros.
  document.getElementById('table-search').addEventListener(
    'input',
    event => {
      filterDataTable(event.target.value);
    }
  );


  // Paginación.
  document.getElementById('prev-page').addEventListener(
    'click',
    () => {
      appState.currentPage -= 1;
      renderDataTable();
    }
  );

  document.getElementById('next-page').addEventListener(
    'click',
    () => {
      appState.currentPage += 1;
      renderDataTable();
    }
  );


  // Selección manual del escenario que se desea comparar.
  document.querySelectorAll('.scenario-btn').forEach(button => {
    button.addEventListener('click', () => {
      loadModelMetrics(button.dataset.scenario);
    });
  });


  // Formulario de predicción.
  document.getElementById('prediction-form').addEventListener(
    'submit',
    submitPrediction
  );

  document.getElementById('prediction-form').addEventListener(
    'reset',
    () => {
      setTimeout(resetPrediction, 0);
    }
  );


  // Carga inicial de los datos originales.
  loadDataset(
    'clientes_solicitud_credito.csv',
    document.querySelector('.source-btn[data-kind="raw"]')
  );


  // Detectar automáticamente el modelo y el escenario en FastAPI.
  // No se fija el escenario a cinco variables.
  checkApi();

});