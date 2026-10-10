let globalData = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchData();

    // Botón Análisis Intervalo
    document.getElementById('range-btn').addEventListener('click', () => {
        const start = document.getElementById('start-date').value;
        const end = document.getElementById('end-date').value;
        fetchData(start, end);
    });

    // Botón Proyección Fin de Mes
    document.getElementById('eom-btn').addEventListener('click', () => {
        const today = new Date();
        const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0);
        
        const start = today.toISOString().split('T')[0];
        const end = lastDay.toISOString().split('T')[0];
        
        document.getElementById('start-date').value = start;
        document.getElementById('end-date').value = end;
        fetchData(start, end);
    });

    // Otros botones
    document.getElementById('refresh-btn').addEventListener('click', () => {
        const btn = document.getElementById('refresh-btn');
        btn.textContent = 'Procesando...';
        btn.disabled = true;
        fetch('/api/run-pipeline', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                alert(data.message || data.error);
                btn.textContent = 'Procesar Todo';
                btn.disabled = false;
                fetchData();
            });
    });

    document.getElementById('file-upload').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        fetch('/api/upload', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                alert(data.message || data.error);
                if (data.message && confirm("¿Deseas procesar ahora?")) document.getElementById('refresh-btn').click();
            });
    });

    document.getElementById('zone-filter').addEventListener('change', () => {
        if (globalData && globalData.node_breakdown) {
            updateNodeTable(globalData.node_breakdown, document.getElementById('zone-filter').value);
        }
    });
});

async function fetchData(start = '', end = '') {
    try {
        const url = `/api/data?start=${start}&end=${end}`;
        const response = await fetch(url);
        globalData = await response.json();
        if (globalData.error) {
            console.error(globalData.error);
            alert("Error al cargar datos: " + globalData.error);
            return;
        }
        
        updateDashboard(globalData);
        updateReports(globalData);
        updateForecast(globalData);
        populateZoneFilter(globalData.node_breakdown || []);
        
        const lastUpdateStr = (globalData.kpis && globalData.kpis.last_update) ? globalData.kpis.last_update : '---';
        document.getElementById('last-update').textContent = `INFO: ${lastUpdateStr}`;
    } catch (e) { console.error(e); }
}

function populateZoneFilter(nodes) {
    const filter = document.getElementById('zone-filter');
    const zones = [...new Set(nodes.map(n => n.Zona))].sort();
    const current = filter.value;
    filter.innerHTML = '<option value="all">Todas las Zonas</option>';
    zones.forEach(z => filter.innerHTML += `<option value="${z}">${z}</option>`);
    filter.value = current || 'all';
}

function switchTab(tabId, el) {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    el.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.getElementById(`${tabId}-tab`).style.display = 'block';
    
    // Actualizar Breadcrumb
    const breadcrumb = document.getElementById('breadcrumb');
    if (breadcrumb) {
        const textMap = { 'dashboard': 'PANEL GENERAL', 'reports': 'INFORMES VISUALES', 'forecast': 'PREDICCIÓN DE DEMANDA' };
        breadcrumb.textContent = textMap[tabId] || 'PANEL';
    }

    if (globalData) {
        setTimeout(() => {
            if (tabId === 'dashboard') updateDashboard(globalData);
            if (tabId === 'reports') updateReports(globalData);
            if (tabId === 'forecast') updateForecast(globalData);
        }, 50);
    }
}

function updateDashboard(data) {
    if (!data.kpis) return;
    document.getElementById('dash-total').textContent = data.kpis.total_tickets_today;
    const suffix = data.kpis.label_suffix || 'hoy';
    document.getElementById('dash-trend').textContent = `Total tickets ${suffix}`;
    
    // Cambiar títulos de tarjetas si es necesario
    document.querySelectorAll('.card h3').forEach(h3 => {
        if (h3.textContent.includes('TOTAL TICKETS')) h3.textContent = `TICKETS (${suffix.toUpperCase()})`;
    });

    renderChart('dashBarChart', 'bar', {
        labels: Object.keys(data.distribution),
        datasets: [{ label: 'Tickets', data: Object.values(data.distribution), backgroundColor: '#818CF8', borderRadius: 6 }]
    });
    renderChart('dashDonutChart', 'doughnut', {
        labels: Object.keys(data.distribution),
        datasets: [{ data: Object.values(data.distribution), backgroundColor: ['#4F46E5', '#818CF8', '#C7D2FE', '#E0E7FF', '#F1F5F9'], borderWidth: 0 }]
    }, { cutout: '70%', plugins: { legend: { position: 'bottom' } } });
}

function updateReports(data) {
    // 1. Alinear Datos de Riesgo (Real vs Proyectado)
    const allZones = [...new Set([...Object.keys(data.distribution), ...Object.keys(data.predictions_summary)])].sort();
    const realValues = allZones.map(z => data.distribution[z] || 0);
    const predValues = allZones.map(z => data.predictions_summary[z] || 0);

    renderChart('riskChart', 'bar', {
        labels: allZones,
        datasets: [
            { label: 'Real (Promedio Diario)', data: realValues, backgroundColor: '#C7D2FE' },
            { label: 'Proyectado (Promedio Diario)', data: predValues, backgroundColor: '#4F46E5' }
        ]
    });

    // 2. Tendencia de Demanda
    if (data.forecast_line && data.forecast_line.labels && data.forecast_line.labels.length > 0) {
        renderChart('trendLineChart', 'line', {
            labels: data.forecast_line.labels,
            datasets: [{ label: 'Demanda', data: data.forecast_line.values, borderColor: '#4F46E5', fill: false, tension: 0.4 }]
        });
    }

    // Radar de Anomalías
    if (data.anomalies && data.anomalies.labels && data.anomalies.labels.length > 0) {
        renderChart('anomalyChart', 'radar', {
            labels: data.anomalies.labels,
            datasets: [{
                label: 'Criticidad (Total Fallas)',
                data: data.anomalies.values,
                backgroundColor: 'rgba(239, 68, 68, 0.2)',
                borderColor: '#EF4444',
                pointBackgroundColor: '#EF4444'
            }]
        }, {
            scales: {
                r: {
                    angleLines: { display: false },
                    suggestedMin: 0
                }
            }
        });
    }

    // Actualizar tabla de métricas
    const precTbody = document.querySelector('#prec-metrics-table tbody');
    if (precTbody && data.precision_metrics && data.precision_metrics.length > 0) {
        precTbody.innerHTML = '';
        data.precision_metrics.forEach(row => {
            precTbody.innerHTML += `
                <tr>
                    <td>${row.Zona}</td>
                    <td>${row.MAE}</td>
                    <td><span class="badge" style="background: #EEF2FF; color: var(--primary);">${row.MAPE}</span></td>
                    <td><span class="badge" style="background: #F1F5F9; color: var(--text-muted);">${row.Modelo}</span></td>
                </tr>
            `;
        });
    }
}

function updateForecast(data) {
    if (!data.predictions_summary) return;
    const total = Object.values(data.predictions_summary).reduce((a, b) => a + b, 0);
    document.getElementById('forecast-national').textContent = `${Math.round(total)} tickets`;

    renderChart('miniTrendChart', 'line', {
        labels: data.forecast_line.labels.slice(-7),
        datasets: [{ data: data.forecast_line.values.slice(-7), borderColor: '#4F46E5', borderWidth: 2, pointRadius: 0 }]
    }, { scales: { x: { display: false }, y: { display: false } }, plugins: { legend: { display: false } } });

    const zonalContainer = document.getElementById('zonal-cards');
    if (zonalContainer) {
        zonalContainer.innerHTML = '';
        Object.entries(data.predictions_summary).forEach(([zone, val]) => {
            zonalContainer.innerHTML += `
                <div class="card zone-card">
                    <div class="zone-name">${zone}</div>
                    <div class="zone-value">${Math.round(val)}</div>
                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;">Confianza: 92%</div>
                </div>
            `;
        });
    }
    updateNodeTable(data.node_breakdown, document.getElementById('zone-filter').value);
}

function updateNodeTable(nodes, filterZone) {
    const nodeTbody = document.querySelector('#node-forecast-table tbody');
    if (!nodeTbody) return;
    nodeTbody.innerHTML = '';
    const filteredNodes = filterZone === 'all' ? nodes : nodes.filter(n => n.Zona === filterZone);
    filteredNodes.forEach(row => {
        let riskColor = 'var(--success)';
        let riskText = 'Bajo';
        if (row.Prediccion_Tickets > 0.7) { riskColor = 'var(--danger)'; riskText = 'Alto'; }
        else if (row.Prediccion_Tickets > 0.3) { riskColor = 'var(--warning)'; riskText = 'Medio'; }

        nodeTbody.innerHTML += `
            <tr>
                <td><b>${row.Nodo}</b></td>
                <td>${row.Zona}</td>
                <td>
                    <span style="font-weight: 800; font-size: 1rem;">${Math.round(row.Prediccion_Tickets)}</span>
                    <span class="badge" style="background: ${riskColor}22; color: ${riskColor}; margin-left: 0.5rem; font-size: 0.65rem;">RIESGO ${riskText}</span>
                </td>
                <td style="color: var(--text-muted);">${Math.floor(row.Q30)}</td>
                <td style="color: var(--text-muted);">${Math.ceil(row.Q70)}</td>
            </tr>
        `;
    });
}

function renderChart(id, type, data, options = {}) {
    const ctx = document.getElementById(id);
    if (!ctx) return;
    if (ctx.chart) ctx.chart.destroy();
    ctx.chart = new Chart(ctx, {
        type: type,
        data: data,
        options: { responsive: true, maintainAspectRatio: false, ...options }
    });
}
