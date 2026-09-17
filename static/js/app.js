/* 
   SCORING CRÉDIT — Interface Web (VERSION CORRIGÉE)
   */

'use strict';

const API_BASE = '';


// 1. UTILITAIRES

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatNumber(n) {
    return Number(n).toLocaleString('fr-FR').replace(/\u202f/g, ' ');
}

function log(...args) {
    console.log('%c[SCORING]', 'color:#3498db;font-weight:bold;', ...args);
}



// 2. NAVIGATION PAR ONGLETS (délégation d'événements)

function activateTab(tabId) {
    log('Activation onglet :', tabId);

    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));

    const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (btn) btn.classList.add('active');

    const section = document.getElementById(tabId);
    if (section) section.classList.add('active');

    if (tabId === 'tab-model') loadModelInfo();
}

function initTabs() {
    const navTabs = document.querySelector('.tabs');
    if (!navTabs) {
        console.warn('⚠️ .tabs introuvable');
        return;
    }

    // Délégation : on écoute sur le parent
    navTabs.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (!btn) return;

        const tabId = btn.dataset.tab;
        if (tabId) activateTab(tabId);
    });

    log('✅ Onglets initialisés');
}



// 3. VÉRIFICATION DE L'API

async function checkApiStatus() {
    const badge = document.getElementById('api-status');
    if (!badge) return;

    try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error('API non disponible');
        const data = await res.json();

        badge.innerHTML = `
            <span class="status-dot"></span>
            <span>API en ligne — v${data.version || '1.0.0'}</span>
        `;
        log('✅ API en ligne');
    } catch (err) {
        badge.innerHTML = `
            <span class="status-dot" style="background:#e74c3c;box-shadow:0 0 10px #e74c3c;"></span>
            <span>API hors ligne</span>
        `;
        console.error('❌ API :', err);
    }
}



// 4. PRÉDICTION INDIVIDUELLE

function getFormData() {
    return {
        age: parseInt($('#age').value, 10),
        revenu_mensuel: parseInt($('#revenu_mensuel').value, 10),
        montant_pret: parseInt($('#montant_pret').value, 10),
        duree_pret_mois: parseInt($('#duree_pret_mois').value, 10),
        taux_interet: parseFloat($('#taux_interet').value),
        nb_emprunts_anterieurs: parseInt($('#nb_emprunts_anterieurs').value, 10),
        score_historique: parseInt($('#score_historique').value, 10),
        situation_familiale: $('#situation_familiale').value,
        secteur_activite: $('#secteur_activite').value,
        garantie: $('#garantie').value,
    };
}

async function submitSinglePrediction(event) {
    event.preventDefault();

    const payload = getFormData();
    const nomClient = $('#nom_client').value || 'Client';
    const btn = $('#btn-predict');
    const resultEl = $('#result-single');

    btn.disabled = true;
    btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> Évaluation...`;

    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || 'Erreur de prédiction');
        }

        const data = await res.json();
        displaySingleResult(nomClient, data, resultEl);

    } catch (err) {
        resultEl.innerHTML = `
            <div class="result-box result-error">
                <div class="result-header">
                    <i class="fas fa-circle-xmark"></i>
                    <div><h3>Erreur</h3><p>${escapeHtml(err.message)}</p></div>
                </div>
            </div>
        `;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fas fa-magnifying-glass-chart"></i> Évaluer le risque`;
    }
}

function displaySingleResult(nomClient, data, container) {
    const isBad = data.defaut === 1;
    const classCard = isBad ? 'result-bad' : 'result-good';
    const icon = isBad ? 'fa-triangle-exclamation' : 'fa-circle-check';
    const statut = isBad ? 'Mauvais client' : 'Bon client';

    let riskClass = 'badge-green';
    if (data.niveau_risque === 'Moyen') riskClass = 'badge-orange';
    if (data.niveau_risque === 'Élevé') riskClass = 'badge-red';

    const probPercent = (data.probabilite_defaut * 100).toFixed(1);

    container.innerHTML = `
        <div class="result-box ${classCard}">
            <div class="result-header">
                <i class="fas ${icon}"></i>
                <div>
                    <h3>${statut}</h3>
                    <p class="result-client">${escapeHtml(nomClient)}</p>
                </div>
            </div>
            <div class="result-metrics">
                <div class="metric">
                    <span class="metric-label">Probabilité de défaut</span>
                    <span class="metric-value">${probPercent}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill" style="width:${probPercent}%; background:${isBad ? '#e74c3c' : '#27ae60'};"></div>
                </div>
                <div class="metric-row">
                    <div class="metric-small">
                        <span class="metric-label">Niveau de risque</span>
                        <span class="badge ${riskClass}">${data.niveau_risque}</span>
                    </div>
                    <div class="metric-small">
                        <span class="metric-label">Ratio d'endettement</span>
                        <span class="metric-value-small">${data.ratio_endettement}</span>
                    </div>
                </div>
            </div>
            <div class="recommendation">
                <i class="fas fa-lightbulb"></i>
                <div>
                    <strong>Recommandation :</strong>
                    <p>${escapeHtml(data.recommandation)}</p>
                </div>
            </div>
        </div>
    `;
}

function resetForm() {
    $('#nom_client').value = 'CLIENT_001';
    $('#age').value = 35;
    $('#revenu_mensuel').value = 250000;
    $('#montant_pret').value = 15000000;
    $('#duree_pret_mois').value = 60;
    $('#taux_interet').value = 8.5;
    $('#nb_emprunts_anterieurs').value = 2;
    $('#score_historique').value = 650;
    $('#situation_familiale').value = 'marié';
    $('#secteur_activite').value = 'commerce';
    $('#garantie').value = 'oui';

    $('#result-single').innerHTML = `
        <div class="empty-state">
            <i class="fas fa-inbox"></i>
            <p>Remplissez le formulaire et cliquez sur <strong>Évaluer le risque</strong></p>
        </div>
    `;
}


// 
// 5. UPLOAD CSV

async function handleCsvUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const csvFilename = $('#csv-filename');
    const csvResult = $('#csv-result');

    csvFilename.innerHTML = `
        <i class="fas fa-file-csv"></i> Fichier chargé : <strong>${escapeHtml(file.name)}</strong>
        <span class="file-size">(${(file.size / 1024).toFixed(1)} Ko)</span>
    `;

    csvResult.innerHTML = `
        <div class="loading"><i class="fas fa-spinner fa-spin"></i> Analyse en cours...</div>
    `;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_BASE}/predict/csv`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || 'Erreur CSV');
        }

        const data = await res.json();
        displayCsvResults(data, file.name);

    } catch (err) {
        csvResult.innerHTML = `
            <div class="result-box result-error">
                <div class="result-header">
                    <i class="fas fa-circle-xmark"></i>
                    <div><h3>Erreur</h3><p>${escapeHtml(err.message)}</p></div>
                </div>
            </div>
        `;
    } finally {
        event.target.value = '';
    }
}

function displayCsvResults(data, filename) {
    const s = data.summary;
    const preds = data.predictions;

    const statsHtml = `
        <div class="stats-grid">
            <div class="stat-card stat-primary">
                <div class="stat-icon"><i class="fas fa-users"></i></div>
                <div class="stat-info"><span class="stat-value">${s.total_clients}</span><span class="stat-label">Clients évalués</span></div>
            </div>
            <div class="stat-card stat-success">
                <div class="stat-icon"><i class="fas fa-circle-check"></i></div>
                <div class="stat-info"><span class="stat-value">${s.bons_clients}</span><span class="stat-label">Bons clients</span></div>
            </div>
            <div class="stat-card stat-danger">
                <div class="stat-icon"><i class="fas fa-triangle-exclamation"></i></div>
                <div class="stat-info"><span class="stat-value">${s.mauvais_clients}</span><span class="stat-label">Mauvais clients</span></div>
            </div>
            <div class="stat-card stat-warning">
                <div class="stat-icon"><i class="fas fa-percent"></i></div>
                <div class="stat-info"><span class="stat-value">${(s.taux_defaut * 100).toFixed(1)}%</span><span class="stat-label">Taux de défaut</span></div>
            </div>
        </div>
    `;

    const riskHtml = `
        <div class="risk-summary">
            <h3><i class="fas fa-chart-pie"></i> Répartition par niveau de risque</h3>
            <div class="risk-bars">
                <div class="risk-bar-item"><span class="badge badge-green">Faible</span><span class="risk-count">${s.risque_faible}</span></div>
                <div class="risk-bar-item"><span class="badge badge-orange">Moyen</span><span class="risk-count">${s.risque_moyen}</span></div>
                <div class="risk-bar-item"><span class="badge badge-red">Élevé</span><span class="risk-count">${s.risque_eleve}</span></div>
            </div>
        </div>
    `;

    const rowsHtml = preds.map((p) => {
        const isBad = p.defaut === 1;
        const rowClass = isBad ? 'row-bad' : 'row-good';
        const badge = isBad
            ? '<span class="badge badge-red">Mauvais client</span>'
            : '<span class="badge badge-green">Bon client</span>';

        return `
            <tr class="${rowClass}">
                <td><strong>${escapeHtml(p.nom_client)}</strong></td>
                <td>${p.age}</td>
                <td>${formatNumber(p.revenu_mensuel)} FCFA</td>
                <td>${formatNumber(p.montant_pret)} FCFA</td>
                <td>${p.score_historique}</td>
                <td>${p.ratio_endettement}</td>
                <td>${(p.probabilite_defaut * 100).toFixed(1)}%</td>
                <td>${badge}</td>
                <td>
                    <span class="badge ${
                        p.niveau_risque === 'Faible' ? 'badge-green'
                        : p.niveau_risque === 'Moyen' ? 'badge-orange'
                        : 'badge-red'
                    }">${p.niveau_risque}</span>
                </td>
            </tr>
        `;
    }).join('');

    const tableHtml = `
        <div class="table-container">
            <div class="table-header">
                <h3><i class="fas fa-table"></i> Détail des prédictions</h3>
                <button class="btn btn-secondary" id="btn-download-csv">
                    <i class="fas fa-download"></i> Télécharger les résultats (CSV)
                </button>
            </div>
            <div class="table-scroll">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Client</th><th>Âge</th><th>Revenu</th><th>Prêt</th>
                            <th>Score</th><th>Ratio</th><th>Prob.</th><th>Statut</th><th>Risque</th>
                        </tr>
                    </thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>
        </div>
    `;

    let errorsHtml = '';
    if (data.errors && data.errors.length > 0) {
        const errRows = data.errors.map(e => `<li>Ligne ${e.ligne} : ${escapeHtml(e.erreur)}</li>`).join('');
        errorsHtml = `
            <div class="errors-box">
                <h4><i class="fas fa-triangle-exclamation"></i> ${data.errors.length} ligne(s) ignorée(s)</h4>
                <ul>${errRows}</ul>
            </div>
        `;
    }

    $('#csv-result').innerHTML = statsHtml + riskHtml + tableHtml + errorsHtml;

    const btnDownload = document.getElementById('btn-download-csv');
    if (btnDownload) {
        btnDownload.addEventListener('click', () => downloadCsvResult(data.csv_result, filename));
    }
}

function downloadCsvResult(csvContent, originalFilename) {
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${originalFilename.replace('.csv', '')}_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


// ============================================================================
// 6. INFORMATIONS DU MODÈLE
// ============================================================================
let modelInfoLoaded = false;

async function loadModelInfo() {
    if (modelInfoLoaded) return;

    const container = document.getElementById('model-info');
    if (!container) return;

    container.innerHTML = `<div class="loading"><i class="fas fa-spinner fa-spin"></i> Chargement...</div>`;

    try {
        const res = await fetch(`${API_BASE}/model/info`);
        if (!res.ok) throw new Error('Impossible de charger les infos');

        const info = await res.json();
        const perf = info.performance;

        container.innerHTML = `
            <div class="info-grid">
                <div class="info-card"><i class="fas fa-brain"></i><h4>Type de modèle</h4><p>${escapeHtml(info.model_type)}</p></div>
                <div class="info-card"><i class="fas fa-tag"></i><h4>Version</h4><p>${escapeHtml(info.version)}</p></div>
                <div class="info-card"><i class="fas fa-bullseye"></i><h4>Accuracy</h4><p>${(perf.accuracy * 100).toFixed(2)}%</p></div>
                <div class="info-card"><i class="fas fa-chart-line"></i><h4>AUC</h4><p>${perf.auc.toFixed(4)}</p></div>
                <div class="info-card"><i class="fas fa-crosshairs"></i><h4>Précision</h4><p>${(perf.precision * 100).toFixed(2)}%</p></div>
                <div class="info-card"><i class="fas fa-magnifying-glass"></i><h4>Rappel</h4><p>${(perf.recall * 100).toFixed(2)}%</p></div>
                <div class="info-card"><i class="fas fa-balance-scale"></i><h4>F1-Score</h4><p>${(perf.f1_score * 100).toFixed(2)}%</p></div>
                <div class="info-card"><i class="fas fa-list"></i><h4>Nb. de features</h4><p>${info.features.length}</p></div>
            </div>

            <div class="features-section">
                <h3><i class="fas fa-cubes"></i> Variables utilisées</h3>
                <div class="features-list">${info.features.map(f => `<span class="feature-tag">${escapeHtml(f)}</span>`).join('')}</div>
            </div>
            <div class="features-section">
                <h3><i class="fas fa-hashtag"></i> Variables numériques</h3>
                <div class="features-list">${info.numerical_features.map(f => `<span class="feature-tag feature-tag-num">${escapeHtml(f)}</span>`).join('')}</div>
            </div>
            <div class="features-section">
                <h3><i class="fas fa-toggle-on"></i> Variables one-hot</h3>
                <div class="features-list">${info.dummy_features.map(f => `<span class="feature-tag feature-tag-dummy">${escapeHtml(f)}</span>`).join('')}</div>
            </div>
        `;

        modelInfoLoaded = true;
    } catch (err) {
        container.innerHTML = `
            <div class="result-box result-error">
                <p><i class="fas fa-circle-xmark"></i> ${escapeHtml(err.message)}</p>
            </div>
        `;
    }
}


// ============================================================================
// 7. INITIALISATION
// ============================================================================
function init() {
    log('🚀 Démarrage');

    log('Éléments trouvés :', {
        tabs: $$('.tab-btn').length,
        contents: $$('.tab-content').length,
        form: !!document.getElementById('form-single'),
    });

    initTabs();
    checkApiStatus();

    const form = document.getElementById('form-single');
    if (form) form.addEventListener('submit', submitSinglePrediction);

    const btnReset = document.getElementById('btn-reset');
    if (btnReset) btnReset.addEventListener('click', resetForm);

    const csvInput = document.getElementById('csv-file');
    if (csvInput) csvInput.addEventListener('change', handleCsvUpload);

    log('✅ Initialisation terminée');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}