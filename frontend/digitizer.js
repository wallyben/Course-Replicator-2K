/**
 * Course Replicator 2K — Manual Digitizer
 *
 * Architecture
 * ────────────
 * • One L.FeatureGroup per layer (display + hit-testing)
 * • One ephemeral L.FeatureGroup (masterEditGroup) used during edit/delete
 *   sessions — features are moved in/out of it so Leaflet.Draw's toolbar
 *   controls operate on a single group.
 * • All persistence goes through the Flask API (/save_layer).
 * • Satellite mosaic loaded as L.imageOverlay aligned to the course bbox.
 *
 * Mode state machine
 * ──────────────────
 *   none  →  draw   : click a layer row in Draw mode
 *   none  →  edit   : click Edit button in mode-bar
 *   none  →  delete : click Delete button in mode-bar
 *   edit / delete → none : click Save or Cancel in session bar
 *   draw  →  none  : press Escape or click another mode button
 */

'use strict';

// ── Layer configuration ───────────────────────────────────────────────────────

const LAYERS = ['greens','fairways','bunkers','water','rough','tees','trees','paths','holes'];

const LAYER_LABEL = {
    greens:   'Greens',
    fairways: 'Fairways',
    bunkers:  'Bunkers',
    water:    'Water Hazards',
    rough:    'Rough Zones',
    tees:     'Tee Boxes',
    trees:    'Tree Clusters',
    paths:    'Cart Paths',
    holes:    'Hole Routes',
};

const LAYER_COLOR = {
    greens:   '#2ecc71',
    fairways: '#27ae60',
    bunkers:  '#e67e22',
    water:    '#3498db',
    rough:    '#8e44ad',
    tees:     '#e74c3c',
    trees:    '#1a5c2a',
    paths:    '#95a5a6',
    holes:    '#f39c12',
};

// Geometry drawn by each layer: polygon | rectangle | polyline
const LAYER_DRAW_TYPE = {
    greens:   'polygon',
    fairways: 'polygon',
    bunkers:  'polygon',
    water:    'polygon',
    rough:    'polygon',
    tees:     'rectangle',
    trees:    'polygon',
    paths:    'polyline',
    holes:    'polyline',
};

const LAYER_FEATURE_TYPE = {
    greens:   'green',
    fairways: 'fairway',
    bunkers:  'bunker',
    water:    'water',
    rough:    'rough',
    tees:     'tee',
    trees:    'trees',
    paths:    'path',
    holes:    'hole',
};

const LAYER_FILL_OPACITY = {
    greens: 0.30, fairways: 0.20, bunkers: 0.50,
    water:  0.40, rough:    0.20, tees:    0.50,
    trees:  0.55, paths:    0.00, holes:   0.00,
};

// ── Global state ──────────────────────────────────────────────────────────────

let map;
let courseMeta          = {};
let featureGroups       = {};     // layerName → L.FeatureGroup
let masterEditGroup     = null;   // temporary group for edit/delete sessions
let currentMode         = 'none'; // 'none'|'draw'|'edit'|'delete'
let activeDrawLayer     = null;   // layer name currently being drawn
let activeDrawHandler   = null;   // L.Draw.* handler instance
let editHandler         = null;   // L.EditToolbar.Edit instance
let deleteHandler       = null;   // L.EditToolbar.Delete instance
let pendingHoleLayer    = null;   // Leaflet layer waiting for hole attribute dialog
let importDetectedData  = {};     // populated when import modal is open
let layerVisible        = {};     // layerName → boolean

// ── UUID generator ────────────────────────────────────────────────────────────

function genUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}

// ── API client ────────────────────────────────────────────────────────────────

const api = {
    async get(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`GET ${url} → ${r.status}`);
        return r.json();
    },
    async post(url, body) {
        const r = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const text = await r.text().catch(() => '');
            throw new Error(`POST ${url} → ${r.status}: ${text}`);
        }
        return r.json();
    },
    getCourse()          { return this.get('/course'); },
    getAllFeatures()      { return this.get('/features'); },
    getCounts()          { return this.get('/counts'); },
    getDetected()        { return this.get('/detected'); },
    saveLayer(layer, features) {
        return this.post('/save_layer', { layer, features });
    },
    clearLayer(layer) {
        return this.post('/clear_layer', { layer });
    },
    exportAll()          { return this.post('/export', {}); },
};

// ── Style builder ─────────────────────────────────────────────────────────────

function layerStyle(layerName) {
    const col = LAYER_COLOR[layerName];
    return {
        color:       col,
        weight:      layerName === 'holes' ? 3 : 2,
        opacity:     0.9,
        fillColor:   col,
        fillOpacity: LAYER_FILL_OPACITY[layerName] || 0.25,
        dashArray:   layerName === 'paths' ? '6 4' : null,
    };
}

// ── Map initialisation ────────────────────────────────────────────────────────

function initMap(meta) {
    const [minLon, minLat, maxLon, maxLat] = meta.bbox;
    const centre = [(minLat + maxLat) / 2, (minLon + maxLon) / 2];

    map = L.map('map', {
        center:    centre,
        zoom:      16,
        zoomSnap:  0.25,
        zoomDelta: 0.5,
        maxZoom:   21,
    });

    // OSM base tiles (reference / fallback)
    const osmTiles = L.tileLayer(
        'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        { maxZoom: 19, opacity: 0.3,
          attribution: '© OpenStreetMap contributors' }
    ).addTo(map);

    // Satellite mosaic overlay
    if (meta.satellite_exists) {
        const bounds = L.latLngBounds(
            [minLat, minLon],
            [maxLat, maxLon]
        );
        L.imageOverlay('/satellite', bounds, {
            opacity:         1.0,
            zIndex:          200,
            interactive:     false,
            crossOrigin:     true,
        }).addTo(map);

        map.fitBounds(bounds, { padding: [20, 20] });
    } else {
        document.getElementById('sat-missing-banner').classList.add('visible');
        map.fitBounds([[minLat, minLon],[maxLat, maxLon]], { padding: [30, 30] });
    }

    // Course boundary overlay (thin dashed white line)
    const bboxPoly = L.rectangle(
        [[minLat, minLon],[maxLat, maxLon]],
        { color: '#ffffff', weight: 1, dashArray: '4 4',
          fill: false, opacity: 0.35, interactive: false }
    ).addTo(map);

    // Keyboard shortcut: Escape → cancel draw / exit mode
    map.getContainer().addEventListener('keydown', e => {
        if (e.key === 'Escape') escapeAction();
    });
    map.getContainer().setAttribute('tabindex', 0);

    // Draw events
    map.on('draw:created',  onDrawCreated);
    map.on('draw:drawstart', () => showHint('Click to add points. Double-click to finish.'));
    map.on('draw:drawstop',  () => hideHint());
}

// ── Feature groups ────────────────────────────────────────────────────────────

function initFeatureGroups() {
    LAYERS.forEach(name => {
        featureGroups[name] = L.featureGroup().addTo(map);
        layerVisible[name]  = true;
    });
}

function getLayerFeatures(layerName) {
    const features = [];
    featureGroups[layerName].eachLayer(l => {
        const f      = l.toGeoJSON();
        f.properties = Object.assign({}, l._extraProps || {}, f.properties);
        if (!f.properties.id) f.properties.id = l._featureId || genUUID();
        features.push(f);
    });
    return features;
}

async function syncLayer(layerName) {
    try {
        const features = getLayerFeatures(layerName);
        await api.saveLayer(layerName, features);
        updateCount(layerName, features.length);
    } catch (e) {
        setStatus(`Auto-save failed: ${e.message}`, 'error');
    }
}

function addLeafletLayerToGroup(layerName, leafletLayer, extraProps) {
    const id            = (extraProps && extraProps.id) || genUUID();
    const props         = Object.assign({ id, type: LAYER_FEATURE_TYPE[layerName], source: 'manual' }, extraProps || {});
    leafletLayer._digitizerLayer = layerName;
    leafletLayer._featureId      = id;
    leafletLayer._extraProps     = props;
    leafletLayer.setStyle(layerStyle(layerName));

    // Popup on click (outside draw/edit/delete mode)
    leafletLayer.on('click', e => {
        if (currentMode !== 'none') return;
        L.DomEvent.stopPropagation(e);
        showFeaturePopup(leafletLayer, layerName, id, e.latlng);
    });

    featureGroups[layerName].addLayer(leafletLayer);
    return id;
}

function showFeaturePopup(leafletLayer, layerName, id, latlng) {
    const props = leafletLayer._extraProps || {};
    let html = `<div class="popup-title">${LAYER_LABEL[layerName]}</div>`;

    if (layerName === 'holes') {
        const hn = props.hole_number ? `Hole ${props.hole_number}` : '';
        const par = props.par ? ` · Par ${props.par}` : '';
        if (hn) html += `<div class="popup-prop">${hn}${par}</div>`;
        if (props.hole_name) html += `<div class="popup-prop">${props.hole_name}</div>`;
    }

    html += `<div class="popup-actions">
        <button class="popup-btn popup-btn-delete"
                onclick="deleteFeatureById('${layerName}','${id}')">
            ✕ Delete
        </button>
    </div>`;

    L.popup({ className: 'digitizer-popup', maxWidth: 200 })
        .setLatLng(latlng)
        .setContent(html)
        .openOn(map);
}

async function deleteFeatureById(layerName, id) {
    map.closePopup();
    featureGroups[layerName].eachLayer(l => {
        if (l._featureId === id) featureGroups[layerName].removeLayer(l);
    });
    await syncLayer(layerName);
    setStatus(`Deleted feature from ${LAYER_LABEL[layerName]}`, 'info');
}
window.deleteFeatureById = deleteFeatureById; // expose for inline HTML onclick

// ── Load all features from server on startup ──────────────────────────────────

async function loadAllFeatures() {
    const all = await api.getAllFeatures();
    LAYERS.forEach(name => {
        const fc = all[name];
        if (!fc || !fc.features || fc.features.length === 0) {
            updateCount(name, 0);
            return;
        }
        loadGeoJSONIntoLayer(name, fc);
        updateCount(name, fc.features.length);
    });
}

function loadGeoJSONIntoLayer(layerName, fc) {
    L.geoJSON(fc, {
        style:         () => layerStyle(layerName),
        onEachFeature: (feature, layer) => {
            const props = feature.properties || {};
            layer._digitizerLayer = layerName;
            layer._featureId      = props.id || genUUID();
            layer._extraProps     = Object.assign({}, props);
            layer.on('click', e => {
                if (currentMode !== 'none') return;
                L.DomEvent.stopPropagation(e);
                showFeaturePopup(layer, layerName, layer._featureId, e.latlng);
            });
            featureGroups[layerName].addLayer(layer);
        },
    });
}

// ── Mode management ───────────────────────────────────────────────────────────

function setMode(mode) {
    if (currentMode === mode) {
        // Toggle off
        setMode('none');
        return;
    }

    // Clean up current mode first
    exitCurrentMode();

    currentMode = mode;

    document.getElementById('btn-mode-draw').classList.toggle('active',
        mode === 'draw');
    document.getElementById('btn-mode-edit').classList.toggle('active-edit',
        mode === 'edit');
    document.getElementById('btn-mode-delete').classList.toggle('active-delete',
        mode === 'delete');

    if (mode === 'draw') {
        setStatus('Select a layer in the panel to start drawing.', 'info');
        highlightLayerRows(true);
    } else if (mode === 'edit') {
        enterEditSession();
    } else if (mode === 'delete') {
        enterDeleteSession();
    } else {
        setStatus('Ready');
        highlightLayerRows(false);
    }

    // Rebuild sidebar to reflect mode
    buildLayerRows();
}
window.setMode = setMode;

function exitCurrentMode() {
    if (activeDrawHandler) {
        activeDrawHandler.disable();
        activeDrawHandler = null;
        activeDrawLayer   = null;
    }
    hideHint();
}

function escapeAction() {
    if (currentMode === 'draw') setMode('none');
}

function highlightLayerRows(on) {
    document.querySelectorAll('.layer-row').forEach(el => {
        if (!on) el.classList.remove('active-draw');
    });
}

// ── Draw tool management ──────────────────────────────────────────────────────

function activateDrawForLayer(layerName) {
    if (currentMode !== 'draw') return;
    if (activeDrawHandler) activeDrawHandler.disable();

    activeDrawLayer = layerName;

    const drawType = LAYER_DRAW_TYPE[layerName];
    const color    = LAYER_COLOR[layerName];
    const shapeOpts = {
        shapeOptions: {
            color:       color,
            fillColor:   color,
            fillOpacity: LAYER_FILL_OPACITY[layerName],
            weight:      2,
        },
        showArea:          false,
        allowIntersection: false,
        repeatMode:        true,   // draw multiple without re-clicking the button
    };

    if (drawType === 'polygon') {
        activeDrawHandler = new L.Draw.Polygon(map, shapeOpts);
    } else if (drawType === 'rectangle') {
        activeDrawHandler = new L.Draw.Rectangle(map, shapeOpts);
    } else {
        activeDrawHandler = new L.Draw.Polyline(map, {
            shapeOptions: {
                color:  color,
                weight: drawType === 'holes' ? 3 : 2,
            },
            repeatMode: true,
        });
    }

    activeDrawHandler.enable();
    showHint(`Drawing ${LAYER_LABEL[layerName]} — click to add points, double-click to finish.`);

    // Highlight the active row
    document.querySelectorAll('.layer-row').forEach(el => {
        el.classList.toggle('active-draw', el.dataset.layer === layerName);
    });
}

async function onDrawCreated(e) {
    const leafletLayer = e.layer;

    if (activeDrawLayer === 'holes') {
        pendingHoleLayer = leafletLayer;
        openHoleModal();
        return;
    }

    if (!activeDrawLayer) return;
    addLeafletLayerToGroup(activeDrawLayer, leafletLayer, {});
    await syncLayer(activeDrawLayer);
    setStatus(`Added to ${LAYER_LABEL[activeDrawLayer]}`, 'ok');
}

// ── Hole attribute modal ──────────────────────────────────────────────────────

function openHoleModal() {
    // Auto-suggest next hole number
    const existing = getLayerFeatures('holes');
    const nums     = existing.map(f => f.properties.hole_number || 0).filter(Boolean);
    const next     = nums.length ? Math.max(...nums) + 1 : 1;
    document.getElementById('modal-hole-num').value  = Math.min(next, 18);
    document.getElementById('modal-par').value       = '4';
    document.getElementById('modal-hole-name').value = '';
    document.getElementById('modal-overlay').classList.add('visible');
    document.getElementById('modal-hole-num').focus();
}

function closeModal() {
    document.getElementById('modal-overlay').classList.remove('visible');
    if (pendingHoleLayer) {
        // User cancelled — discard the drawn shape
        pendingHoleLayer = null;
    }
}
window.closeModal = closeModal;

async function confirmHoleModal() {
    if (!pendingHoleLayer) { closeModal(); return; }

    const holeNum  = parseInt(document.getElementById('modal-hole-num').value, 10);
    const par      = parseInt(document.getElementById('modal-par').value, 10);
    const holeName = document.getElementById('modal-hole-name').value.trim();

    closeModal();

    addLeafletLayerToGroup('holes', pendingHoleLayer, {
        hole_number: holeNum,
        par,
        hole_name:   holeName || undefined,
    });
    pendingHoleLayer = null;
    await syncLayer('holes');
    setStatus(`Hole ${holeNum} (Par ${par}) added.`, 'ok');
}
window.confirmHoleModal = confirmHoleModal;

// ── Edit session (vertex editing) ─────────────────────────────────────────────

function enterEditSession() {
    masterEditGroup = L.featureGroup().addTo(map);

    // Move all visible features into masterEditGroup
    LAYERS.forEach(name => {
        featureGroups[name].eachLayer(l => {
            featureGroups[name].removeLayer(l);
            masterEditGroup.addLayer(l);
        });
    });

    editHandler = new L.EditToolbar.Edit(map, {
        featureGroup: masterEditGroup,
        poly: { allowIntersection: false },
    });
    editHandler.enable();

    showSessionBar('Edit mode — drag vertices to reshape features.');
    setStatus('Editing: drag vertices to reshape. Press Save when done.', 'info');
}

function enterDeleteSession() {
    masterEditGroup = L.featureGroup().addTo(map);

    LAYERS.forEach(name => {
        featureGroups[name].eachLayer(l => {
            featureGroups[name].removeLayer(l);
            masterEditGroup.addLayer(l);
        });
    });

    deleteHandler = new L.EditToolbar.Delete(map, {
        featureGroup: masterEditGroup,
    });
    deleteHandler.enable();

    showSessionBar('Delete mode — click features to mark for deletion.');
    setStatus('Delete mode: click features to mark them, then press Save.', 'info');
}

async function saveSession() {
    if (currentMode === 'edit' && editHandler) {
        editHandler.save();
        editHandler.disable();
        editHandler = null;
    }
    if (currentMode === 'delete' && deleteHandler) {
        deleteHandler.save();
        deleteHandler.disable();
        deleteHandler = null;
    }

    await redistributeFromMasterGroup();
    hideSessionBar();

    // Sync all layers (features may have moved between groups or been deleted)
    setStatus('Saving changes…', 'info');
    for (const name of LAYERS) {
        await syncLayer(name);
    }

    currentMode = 'none';
    clearModeBtns();
    buildLayerRows();
    setStatus('Changes saved.', 'ok');
}
window.saveSession = saveSession;

async function cancelSession() {
    if (currentMode === 'edit' && editHandler) {
        editHandler.revertLayers();
        editHandler.disable();
        editHandler = null;
    }
    if (currentMode === 'delete' && deleteHandler) {
        deleteHandler.revertLayers();
        deleteHandler.disable();
        deleteHandler = null;
    }

    await redistributeFromMasterGroup();
    hideSessionBar();

    currentMode = 'none';
    clearModeBtns();
    buildLayerRows();
    setStatus('Edit cancelled.', 'warn');
}
window.cancelSession = cancelSession;

async function redistributeFromMasterGroup() {
    if (!masterEditGroup) return;

    const toRedistribute = masterEditGroup.getLayers().slice();
    masterEditGroup.clearLayers();
    masterEditGroup.remove();
    masterEditGroup = null;

    toRedistribute.forEach(l => {
        const layerName = l._digitizerLayer;
        if (layerName && featureGroups[layerName]) {
            featureGroups[layerName].addLayer(l);
        }
    });
}

function clearModeBtns() {
    document.getElementById('btn-mode-draw').classList.remove('active');
    document.getElementById('btn-mode-edit').classList.remove('active-edit');
    document.getElementById('btn-mode-delete').classList.remove('active-delete');
}

// ── Layer visibility toggle ───────────────────────────────────────────────────

function toggleLayerVisibility(layerName) {
    const vis = !layerVisible[layerName];
    layerVisible[layerName] = vis;

    if (vis) {
        if (!map.hasLayer(featureGroups[layerName])) {
            featureGroups[layerName].addTo(map);
        }
    } else {
        if (map.hasLayer(featureGroups[layerName])) {
            map.removeLayer(featureGroups[layerName]);
        }
    }

    buildLayerRows();
}
window.toggleLayerVisibility = toggleLayerVisibility;

// ── Sidebar builder ───────────────────────────────────────────────────────────

function buildLayerRows() {
    const panel = document.getElementById('layers-panel');
    panel.innerHTML = '';

    const sections = [
        { title: 'Playing Surfaces', layers: ['greens','fairways','tees','rough'] },
        { title: 'Hazards',          layers: ['bunkers','water'] },
        { title: 'Landscape',        layers: ['trees','paths'] },
        { title: 'Hole Routing',     layers: ['holes'] },
    ];

    sections.forEach(({ title, layers }) => {
        const sec = document.createElement('div');
        sec.className = 'layer-section';

        const titleEl = document.createElement('div');
        titleEl.className = 'layer-section-title';
        titleEl.textContent = title;
        sec.appendChild(titleEl);

        layers.forEach(name => {
            const row     = document.createElement('div');
            row.className = 'layer-row';
            row.dataset.layer = name;

            if (currentMode === 'draw' && activeDrawLayer === name) {
                row.classList.add('active-draw');
            }

            const dot = document.createElement('div');
            dot.className   = 'layer-dot';
            dot.style.background = LAYER_COLOR[name];

            const label = document.createElement('div');
            label.className   = 'layer-label';
            label.textContent = LAYER_LABEL[name];

            const count = document.createElement('div');
            count.className  = 'layer-count';
            count.id         = `count-${name}`;
            const n = featureGroups[name] ? featureGroups[name].getLayers().length : 0;
            count.textContent = n;
            if (n > 0) count.classList.add('has-features');

            const visBtn = document.createElement('button');
            visBtn.className = 'layer-vis-toggle' + (layerVisible[name] ? ' visible' : '');
            visBtn.title     = layerVisible[name] ? 'Hide layer' : 'Show layer';
            visBtn.textContent = layerVisible[name] ? '👁' : '○';
            visBtn.onclick = ev => {
                ev.stopPropagation();
                toggleLayerVisibility(name);
            };

            row.appendChild(dot);
            row.appendChild(label);
            row.appendChild(count);
            row.appendChild(visBtn);

            if (currentMode === 'draw') {
                row.onclick = () => activateDrawForLayer(name);
                row.style.cursor = 'pointer';
            }

            sec.appendChild(row);
        });

        panel.appendChild(sec);
    });
}

function updateCount(layerName, count) {
    const el = document.getElementById(`count-${layerName}`);
    if (!el) return;
    el.textContent = count;
    el.classList.toggle('has-features', count > 0);
}

// ── Import detected features ──────────────────────────────────────────────────

async function openImportModal() {
    setStatus('Loading auto-detected features…', 'info');
    try {
        importDetectedData = await api.getDetected();
    } catch (e) {
        setStatus(`Failed to load detected features: ${e.message}`, 'error');
        return;
    }

    const checksDiv = document.getElementById('import-checks');
    checksDiv.innerHTML = '';

    const available = Object.keys(importDetectedData);
    if (available.length === 0) {
        checksDiv.innerHTML = '<p style="font-size:11px;color:#8b949e;">No auto-detected features found in output directory.</p>';
    } else {
        available.forEach(layer => {
            const fc    = importDetectedData[layer];
            const count = fc && fc.features ? fc.features.length : 0;
            if (count === 0) return;

            const row = document.createElement('div');
            row.className = 'import-check-row';
            row.innerHTML = `
                <input type="checkbox" id="imp-${layer}" value="${layer}" checked>
                <label for="imp-${layer}" style="cursor:pointer">
                    <span style="color:${LAYER_COLOR[layer]};font-weight:600">${LAYER_LABEL[layer]}</span>
                    <span style="color:#484f58"> (${count} features)</span>
                </label>`;
            checksDiv.appendChild(row);
        });
    }

    document.getElementById('import-modal-overlay').style.display = 'flex';
    setStatus('Ready');
}
window.openImportModal = openImportModal;

function closeImportModal() {
    document.getElementById('import-modal-overlay').style.display = 'none';
    importDetectedData = {};
}
window.closeImportModal = closeImportModal;

async function confirmImport() {
    const boxes = document.querySelectorAll('#import-checks input[type=checkbox]:checked');
    const selected = Array.from(boxes).map(b => b.value);
    closeImportModal();

    if (selected.length === 0) { setStatus('No layers selected.', 'warn'); return; }

    setStatus(`Importing ${selected.length} layer(s)…`, 'info');

    for (const layerName of selected) {
        const fc = importDetectedData[layerName];
        if (!fc || !fc.features) continue;

        // Clear existing features in this layer
        featureGroups[layerName].clearLayers();

        // Load into map
        loadGeoJSONIntoLayer(layerName, fc);

        // Persist to store
        const features = getLayerFeatures(layerName);
        await api.saveLayer(layerName, features);
        updateCount(layerName, features.length);
    }

    buildLayerRows();
    setStatus(`Imported ${selected.length} layer(s) from auto-detection.`, 'ok');
}
window.confirmImport = confirmImport;

// ── Export to pipeline ────────────────────────────────────────────────────────

async function exportToPipeline() {
    const btn = document.getElementById('btn-export');
    btn.disabled = true;
    btn.textContent = 'Exporting…';
    setStatus('Exporting to pipeline output directory…', 'info');

    try {
        const result  = await api.exportAll();
        const summary = result.summary || {};

        const lines = Object.entries(summary)
            .filter(([, n]) => n > 0)
            .map(([layer, n]) => `${LAYER_LABEL[layer]}: ${n}`)
            .join('\n');

        const resEl = document.getElementById('export-result');
        resEl.textContent = `Exported ${result.total} features\n${lines}`;
        resEl.classList.add('visible');

        setStatus(`Export complete — ${result.total} features written.`, 'ok');
    } catch (e) {
        setStatus(`Export failed: ${e.message}`, 'error');
    } finally {
        btn.disabled    = false;
        btn.textContent = '⬆ Export to Pipeline';
    }
}
window.exportToPipeline = exportToPipeline;

// ── UI helpers ────────────────────────────────────────────────────────────────

function setStatus(msg, type = '') {
    const el = document.getElementById('status-bar');
    el.textContent = msg;
    el.className   = type ? `status-bar ${type}` : '';
    el.id          = 'status-bar'; // preserve id
}

function showHint(msg) {
    const el = document.getElementById('map-hint');
    el.textContent = msg;
    el.classList.add('visible');
}

function hideHint() {
    document.getElementById('map-hint').classList.remove('visible');
}

function showSessionBar(label) {
    const bar = document.getElementById('session-bar');
    document.getElementById('session-label').textContent = label;
    bar.classList.add('visible');
}

function hideSessionBar() {
    document.getElementById('session-bar').classList.remove('visible');
}

// ── Initialisation ────────────────────────────────────────────────────────────

async function init() {
    setStatus('Connecting to server…', 'info');
    try {
        courseMeta = await api.getCourse();

        document.getElementById('course-name').textContent =
            courseMeta.name || 'Unknown Course';
        document.getElementById('course-area').textContent =
            courseMeta.area_ha ? `${courseMeta.area_ha} ha` : '';

        initMap(courseMeta);
        initFeatureGroups();
        buildLayerRows();

        setStatus('Loading features…', 'info');
        await loadAllFeatures();
        buildLayerRows();

        setStatus('Ready — select Draw mode and click a layer to begin.', 'info');
    } catch (e) {
        setStatus(`Init error: ${e.message}`, 'error');
        console.error('[digitizer] init error:', e);
    }
}

window.addEventListener('load', init);
