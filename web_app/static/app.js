document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ─────────────────────────────────────────────────────────────
    const fileInput = document.getElementById('file-input');
    const btnUpload = document.getElementById('btn-upload');
    const fileList = document.getElementById('file-list');

    const wrapper = document.getElementById('wrapper');
    const wrapperSide = document.getElementById('wrapper-side');
    const imgSingle = document.getElementById('img-single');
    const imgOrigSide = document.getElementById('img-orig-side');
    const imgSegSide = document.getElementById('img-seg-side');

    const currentName = document.getElementById('current-name');
    const metricTime = document.getElementById('metric-time');
    const metricFps = document.getElementById('metric-fps');
    const metricPolygons = document.getElementById('metric-polygons');

    const inputLabelsDir = document.getElementById('input-labels-dir');
    const btnPickLabelsYaml = document.getElementById('btn-pick-labels-yaml');
    const photoMetrics = document.getElementById('photo-metrics-panel');
    // Dataset metrics panel elements
    const pmGtBadge   = document.getElementById('pm-gt-badge');
    const pmAvgTime   = document.getElementById('pm-avg-time');
    const pmAvgTime20 = document.getElementById('pm-avg-time-20');
    const pmIou       = document.getElementById('pm-iou');
    const pmPrec      = document.getElementById('pm-precision');
    const pmRecall    = document.getElementById('pm-recall');
    const pmF1        = document.getElementById('pm-f1');
    const pmCounts    = document.getElementById('pm-counts');

    const videoScrubberContainer = document.getElementById('video-scrubber-container');
    const btnPlayPause = document.getElementById('btn-play-pause');
    const videoTimeEl = document.getElementById('video-time');
    const videoSlider = document.getElementById('video-slider');

    const lmStatus = document.getElementById('lm-status');
    const lmIouAvg = document.getElementById('lm-iou-avg');
    const lmIouMin = document.getElementById('lm-iou-min');
    const lmJitter = document.getElementById('lm-jitter');
    const lmFlicker = document.getElementById('lm-flicker');

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const imgArea = document.getElementById('img-area');
    const dropZone = document.getElementById('drop-zone');
    const chkMetricsOnly      = document.getElementById('chk-metrics-only');
    const metricsOnlyRow      = document.getElementById('metrics-only-row');
    const metricsOnlyLockBadge = document.getElementById('metrics-only-lock-badge');
    const metricsOnlyPlaceholder = document.getElementById('metrics-only-placeholder');

    const modelSelect = document.getElementById('model-select');
    const modelLoading = document.getElementById('model-loading');

    // ── Load Models ──────────────────────────────────────────────────────────
    async function fetchModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            modelSelect.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                modelSelect.appendChild(opt);
            });
            if (data.current) modelSelect.value = data.current;
        } catch (e) { console.error("Error loading models", e); }
    }
    fetchModels();

    modelSelect.addEventListener('change', async (e) => {
        const newModelId = e.target.value;
        modelSelect.disabled = true;
        modelLoading.style.display = 'inline';
        try {
            const fd = new FormData();
            fd.append('model_id', newModelId);
            const res = await fetch('/api/set_model', { method: 'POST', body: fd });
            const data = await res.json();
            if (!data.success) {
                alert("Помилка завантаження моделі: " + data.error);
                await fetchModels(); // revert
            } else {
                // Reset all photos and re-process queue to update global metrics
                filesData.forEach(f => {
                    if (!f.isVideo) {
                        f.processed = false;
                        markDone(f.id, '⏳', 'Очікує');
                    }
                });
                processQueue();
                if (currentFileId) {
                    const fd = filesData.find(f => f.id === currentFileId);
                    if (fd && fd.isVideo) {
                        startVideoStream(typeof isVideoPaused !== 'undefined' ? isVideoPaused : false);
                    }
                    renderMainView();
                }
            }
        } catch (err) {
            console.error(err);
        } finally {
            modelSelect.disabled = false;
            modelLoading.style.display = 'none';
        }
    });

    // ── State ────────────────────────────────────────────────────────────────
    let filesData = [];
    let currentFileId = null;
    let currentFilter = 'all';
    let currentViewMode = 'side';   // default: side-by-side
    let currentGtView = 'pred';
    let currentWs = null;
    let totalFrames = 0;
    let isDraggingSlider = false;
    let isVideoPaused = true;     // default: paused
    let isProcessing = false;
    let metricsOnlyMode = false;

    // ── Pill toggle groups ───────────────────────────────────────────────────
    function setupPillGroup(groupId, onSelect) {
        const group = document.getElementById(groupId);
        if (!group) return;
        group.addEventListener('click', (e) => {
            const btn = e.target.closest('.pill-btn');
            if (!btn) return;
            group.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            onSelect(btn.dataset.value);
        });
    }

    // Set initial active states in the DOM
    setPillActive('filterGroup', currentFilter);
    setPillActive('modeGroup', currentViewMode);
    setPillActive('gtToggleGroup', currentGtView);

    setupPillGroup('filterGroup', (val) => {
        currentFilter = val;
        if (currentWs && currentWs.readyState === WebSocket.OPEN) {
            currentWs.send(JSON.stringify({ command: 'set_filter', filter: currentFilter }));
        }
        renderMainView();
    });

    setupPillGroup('modeGroup', (val) => {
        currentViewMode = val;
        renderMainView();
    });

    setupPillGroup('gtToggleGroup', (val) => {
        currentGtView = val;
        renderMainView();
    });

    function setPillActive(groupId, value) {
        const group = document.getElementById(groupId);
        if (!group) return;
        group.querySelectorAll('.pill-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.value === value);
        });
    }

    // Space: toggle segmented ↔ original
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.code === 'Space') {
            e.preventDefault();
            currentViewMode = currentViewMode === 'segmented' ? 'original' : 'segmented';
            setPillActive('modeGroup', currentViewMode);
            renderMainView();
        }
    });

    // ── Folder / File picker ───────────────────────────────────────────────
    async function pickFolder() {
        const res = await fetch('/api/pick_folder');
        const data = await res.json();
        return data.path || '';
    }

    async function pickFile() {
        const res = await fetch('/api/pick_file');
        const data = await res.json();
        return data.path || '';
    }

    async function onLabelsPathPicked(path) {
        if (!path) return;
        inputLabelsDir.value = path;
        onLabelsPathChanged();
    }

    async function onLabelsPathChanged() {
        const path = inputLabelsDir.value.trim();
        // If it's a video, notify the websocket
        const fd = filesData.find(f => f.id === currentFileId);
        if (!fd) return;
        if (fd.isVideo) {
            if (currentWs && currentWs.readyState === WebSocket.OPEN) {
                currentWs.send(JSON.stringify({ command: 'set_labels_dir', path }));
            }
        } else if (fd.processed) {
            // Re-evaluate current photo
            await reprocessPhotoMetrics(fd);
            renderMainView();
        }
    }

    btnPickLabelsYaml.addEventListener('click', async () => onLabelsPathPicked(await pickFile()));
    inputLabelsDir.addEventListener('change', onLabelsPathChanged);
    inputLabelsDir.addEventListener('input', onLabelsPathChanged);

    // ── Metrics-Only toggle ──────────────────────────────────────────────────
    chkMetricsOnly.addEventListener('change', () => {
        metricsOnlyMode = chkMetricsOnly.checked;
    });

    function lockToggle() {
        chkMetricsOnly.disabled = true;
        metricsOnlyRow.classList.add('is-locked');
        if (metricsOnlyLockBadge) {
            metricsOnlyLockBadge.style.display = 'inline';
            metricsOnlyLockBadge.textContent = metricsOnlyMode ? '🔒 Активно' : '🔒 Режим обрано';
        }
    }

    function unlockToggle() {
        chkMetricsOnly.disabled = false;
        metricsOnlyRow.classList.remove('is-locked');
        if (metricsOnlyLockBadge) metricsOnlyLockBadge.style.display = 'none';
    }

    // ── File Upload & Drag'n'Drop ────────────────────────────────────────────
    btnUpload.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    // Prevent browser from opening dropped files natively
    window.addEventListener('dragover', (e) => e.preventDefault(), false);
    window.addEventListener('drop', (e) => e.preventDefault(), false);

    // Visual drop zone
    document.addEventListener('dragover', () => dropZone.classList.add('active'));
    document.addEventListener('dragleave', (e) => {
        if (!e.relatedTarget || e.relatedTarget === document.documentElement) {
            dropZone.classList.remove('active');
        }
    });
    document.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        if (e.dataTransfer && e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });

    async function handleFiles(files) {
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const isVid = file.type.startsWith('video/');
            if (!file.type.startsWith('image/') && !isVid) continue;

            const id = Date.now() + '_' + i;
            // In metrics-only mode skip base64 reading to save RAM
            const originalB64 = (isVid || metricsOnlyMode) ? null : await readAsDataURL(file);

            filesData.push({
                id, file, isVideo: isVid,
                serverPath: null, originalB64,
                segmentedB64: null, segmentedMasksB64: null, segmentedBoxesB64: null,
                timeMs: null, polygons: 0, processed: false,
                metrics: null,
                metricsOnly: metricsOnlyMode  // remember mode at the time of upload
            });
            addToSidebar(id, file.name, isVid);
        }
        // Lock the toggle once processing starts
        lockToggle();
        processQueue();
    }

    function readAsDataURL(file) {
        return new Promise(res => {
            const r = new FileReader();
            r.onload = e => res(e.target.result);
            r.readAsDataURL(file);
        });
    }

    function addToSidebar(id, name, isVideo) {
        const div = document.createElement('div');
        div.className = 'file-item';
        div.id = `file-${id}`;
        div.innerHTML = `<span>${isVideo ? '🎬' : '🖼️'}</span>
                         <span class="file-name" title="${name}">${name}</span>
                         <span class="file-ok">⏳</span>`;
        div.addEventListener('click', () => selectFile(id));
        fileList.appendChild(div);
        if (filesData.length === 1) selectFile(id);
    }

    // ── Processing queue ─────────────────────────────────────────────────────
    async function processQueue() {
        if (isProcessing) return;
        if (!filesData.some(f => !f.processed)) return;

        isProcessing = true;
        progressContainer.style.display = 'block';

        for (let i = 0; i < filesData.length; i++) {
            if (filesData[i].processed) continue;
            updateProgress(filesData.filter(f => f.processed).length, filesData.length);

            try {
                if (filesData[i].isVideo) {
                    const res = await fetch('/api/upload_video?filename=' + encodeURIComponent(filesData[i].file.name), {
                        method: 'POST', body: filesData[i].file,
                        headers: { 'Content-Type': filesData[i].file.type || 'application/octet-stream' }
                    });
                    const data = await res.json();
                    if (data.success) {
                        filesData[i].serverPath = data.path;
                        filesData[i].processed = true;
                        markDone(filesData[i].id, '🎬');
                        if (currentFileId === filesData[i].id) renderMainView();
                    }
                } else {
                    const fd2 = new FormData();
                    fd2.append('file', filesData[i].file);
                    const labelsDir = inputLabelsDir.value.trim();
                    if (labelsDir) fd2.append('labels_dir', labelsDir);
                    // Metrics-only: tell server to skip encoding images
                    if (filesData[i].metricsOnly) fd2.append('no_images', 'true');
                    const res = await fetch('/api/process', { method: 'POST', body: fd2 });
                    const data = await res.json();
                    if (data.success) {
                        Object.assign(filesData[i], {
                            // Only store images if NOT metrics-only (saves RAM)
                            segmentedB64:      filesData[i].metricsOnly ? null : data.image,
                            segmentedMasksB64: filesData[i].metricsOnly ? null : data.image_masks,
                            segmentedBoxesB64: filesData[i].metricsOnly ? null : data.image_boxes,
                            gtB64:             filesData[i].metricsOnly ? null : data.image_gt,
                            timeMs:   data.time_ms,
                            polygons: data.polygons,
                            processed: true,
                            metrics: data.metrics || null
                        });
                        
                        let icon = filesData[i].metricsOnly ? '📊' : '✅';
                        let titleText = 'Оброблено успішно';
                        
                        markDone(filesData[i].id, icon, titleText);
                        updateBatchMetrics();
                        if (currentFileId === filesData[i].id) renderMainView();
                    }
                }
            } catch (err) {
                console.error(err);
                markDone(filesData[i].id, '❌');
            }
        }

        updateProgress(filesData.length, filesData.length);
        setTimeout(() => { progressContainer.style.display = 'none'; }, 1500);
        isProcessing = false;
        // Toggle remains locked for the entire session — mode was chosen at upload time
    }

    function markDone(id, icon, titleText = '') {
        const el = document.getElementById(`file-${id}`);
        if (!el) return;
        el.classList.add('processed');
        const s = el.querySelector('.file-ok');
        if (s) {
            s.textContent = icon;
            if (titleText) s.title = titleText;
        }
    }

    function updateProgress(done, total) {
        progressText.textContent = `${done} / ${total}`;
        progressBar.style.width = `${(done / total) * 100}%`;
    }

    // ── Dataset Metrics (aggregated over all processed photos) ───────────────
    function updateBatchMetrics() {
        const photos = filesData.filter(f => !f.isVideo && f.processed);
        if (photos.length === 0) {
            photoMetrics.style.display = 'none';
            return;
        }

        photoMetrics.style.display = 'block';

        // ── Algorithm speed (pure inference time, no extra overhead) ──────────
        const times = photos
            .filter(f => f.timeMs != null && !isNaN(f.timeMs))
            .map(f => Number(f.timeMs));
        if (times.length > 0) {
            const avgAll = times.reduce((a, b) => a + b, 0) / times.length;
            const count20 = Math.max(1, Math.floor(times.length * 0.2));
            const avg20   = times.slice(-count20).reduce((a, b) => a + b, 0) / count20;
            if (pmAvgTime)   pmAvgTime.textContent   = `${avgAll.toFixed(2)} ms`;
            if (pmAvgTime20) pmAvgTime20.textContent = `${avg20.toFixed(2)} ms`;
        } else {
            if (pmAvgTime)   pmAvgTime.textContent   = '—';
            if (pmAvgTime20) pmAvgTime20.textContent = '—';
        }

        // ── Quality metrics (only photos that have GT) ────────────────────────
        const withGt = photos.filter(f => f.metrics && f.metrics.has_gt);
        if (withGt.length > 0) {
            const avg = key => withGt.reduce((s, f) => s + f.metrics[key], 0) / withGt.length;
            if (pmIou)    pmIou.textContent    = avg('iou').toFixed(3);
            if (pmPrec)   pmPrec.textContent   = avg('precision').toFixed(3);
            if (pmRecall) pmRecall.textContent = avg('recall').toFixed(3);
            if (pmF1)     pmF1.textContent     = avg('f1').toFixed(3);
            if (pmGtBadge) {
                pmGtBadge.textContent = `✅ GT: ${withGt.length} / ${photos.length}`;
                pmGtBadge.className   = 'gt-badge gt-yes';
            }
            if (pmCounts) {
                const totalPred = withGt.reduce((s, f) => s + f.metrics.objects, 0);
                const totalGt   = withGt.reduce((s, f) => s + f.metrics.gt_count, 0);
                pmCounts.textContent = `Pred: ${totalPred}  |  GT: ${totalGt}`;
            }
        } else {
            if (pmIou)    pmIou.textContent    = '—';
            if (pmPrec)   pmPrec.textContent   = '—';
            if (pmRecall) pmRecall.textContent = '—';
            if (pmF1)     pmF1.textContent     = '—';
            if (pmGtBadge) {
                pmGtBadge.textContent = '❌ без GT';
                pmGtBadge.className   = 'gt-badge gt-no';
            }
            if (pmCounts) pmCounts.textContent = 'GT-файли не знайдено';
        }
    }

    // ── Re-evaluate photo metrics with a new labels dir ──────────────────
    async function reprocessPhotoMetrics(fdEntry) {
        await reprocessPhoto(fdEntry);
    }

    async function reprocessPhoto(fdEntry) {
        const labelsDir = inputLabelsDir.value.trim();
        if (fdEntry.isVideo) return;
        try {
            const fd2 = new FormData();
            fd2.append('file', fdEntry.file);
            if (labelsDir) fd2.append('labels_dir', labelsDir);
            if (fdEntry.metricsOnly) fd2.append('no_images', 'true');
            const res = await fetch('/api/process', { method: 'POST', body: fd2 });
            const data = await res.json();
            if (data.success) {
                Object.assign(fdEntry, {
                    segmentedB64:      fdEntry.metricsOnly ? null : data.image,
                    segmentedMasksB64: fdEntry.metricsOnly ? null : data.image_masks,
                    segmentedBoxesB64: fdEntry.metricsOnly ? null : data.image_boxes,
                    gtB64:             fdEntry.metricsOnly ? null : data.image_gt,
                    timeMs:   data.time_ms,
                    polygons: data.polygons,
                    processed: true,
                    metrics: data.metrics || null
                });
                let icon = fdEntry.metricsOnly ? '📊' : '✅';
                markDone(fdEntry.id, icon, 'Оброблено успішно');
                updateBatchMetrics();
            }
        } catch (e) { console.error(e); }
    }

    // ── Video scrubber ───────────────────────────────────────────────────────
    videoSlider.addEventListener('mousedown', () => { isDraggingSlider = true; });
    videoSlider.addEventListener('mouseup', () => { isDraggingSlider = false; });
    videoSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        videoTimeEl.textContent = `${val} / ${totalFrames}`;
        if (currentWs && currentWs.readyState === WebSocket.OPEN) {
            currentWs.send(JSON.stringify({ command: 'seek', frame: val }));
        }
    });

    btnPlayPause.addEventListener('click', () => {
        if (!currentWs || currentWs.readyState !== WebSocket.OPEN) return;
        isVideoPaused = !isVideoPaused;
        btnPlayPause.textContent = isVideoPaused ? '▶️' : '⏸️';
        currentWs.send(JSON.stringify({ command: 'pause', state: isVideoPaused }));
    });

    // ── Live stream ──────────────────────────────────────────────────────────
    function resetLiveMetrics() {
        lmStatus.textContent = 'Без GT';
        lmStatus.className = 'lm-status';
        lmIouAvg.textContent = '—';
        lmIouMin.textContent = '—';
        lmJitter.textContent = '—';
        lmFlicker.textContent = '—';
    }

    function startVideoStream(startPaused) {
        const fd = filesData.find(f => f.id === currentFileId);
        if (!fd || !fd.isVideo || !fd.serverPath) return;
        if (currentWs) return; // already streaming

        isVideoPaused = startPaused !== false; // default: paused
        btnPlayPause.textContent = isVideoPaused ? '▶️' : '⏸️';

        const wsUrl = `ws://${window.location.host}/ws/video?path=${encodeURIComponent(fd.serverPath)}`;
        currentWs = new WebSocket(wsUrl);

        currentWs.onopen = () => {
            videoScrubberContainer.style.display = 'flex';
            resetLiveMetrics();
        };

        currentWs.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'init') {
                totalFrames = data.total_frames;
                videoSlider.max = totalFrames;
                videoTimeEl.textContent = `0 / ${totalFrames}`;
                
                const ws = event.target;
                try {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ command: 'set_filter', filter: currentFilter }));
                        ws.send(JSON.stringify({ command: 'set_labels_dir', path: inputLabelsDir.value.trim() }));
                        if (isVideoPaused) {
                            ws.send(JSON.stringify({ command: 'pause', state: true }));
                        }
                    }
                } catch (err) {
                    console.error("Error sending init commands:", err);
                }
                return;
            }

            if (data.type === 'done') { currentWs.close(); return; }
            if (data.error) { alert(data.error); return; }

            if (data.type === 'frame') {
                if (currentViewMode === 'side') {
                    imgOrigSide.src = data.image_orig;
                    imgSegSide.src = data.image;
                } else if (currentViewMode === 'original') {
                    imgSingle.src = data.image_orig;
                } else {
                    imgSingle.src = data.image;
                }

                metricFps.style.display = 'inline-flex';
                metricFps.textContent = `⚡ ${data.fps} FPS`;
                metricTime.style.display = 'inline-flex';
                metricTime.textContent = `⏱️ ${data.time_ms}ms`;

                if (!isDraggingSlider) {
                    videoSlider.value = data.frame_idx;
                    videoTimeEl.textContent = `${data.frame_idx} / ${totalFrames}`;
                }

                const m = data.metrics;
                if (!m) { resetLiveMetrics(); return; }

                if (m.warming_up) {
                    lmStatus.textContent = 'Warming Up…';
                    lmStatus.className = 'lm-status warming';
                    lmIouAvg.textContent = '—'; lmIouMin.textContent = '—';
                    lmJitter.textContent = '—'; lmFlicker.textContent = '—';
                } else if (m.has_data) {
                    lmStatus.textContent = 'Знайдено GT / Рахую';
                    lmStatus.className = 'lm-status running';
                    document.getElementById('lm-iou-cur').textContent = m.iou_cur.toFixed(3);
                    lmIouAvg.textContent = m.iou_avg.toFixed(3);
                    lmIouMin.textContent = m.iou_min.toFixed(3);
                    lmJitter.textContent = m.jitter.toFixed(1);
                    lmFlicker.textContent = m.flicker.toFixed(1) + '%';
                } else {
                    lmStatus.textContent = 'Без GT';
                    lmStatus.className = 'lm-status';
                    document.getElementById('lm-iou-cur').textContent = '—';
                    lmIouAvg.textContent = '—'; lmIouMin.textContent = '—';
                    lmJitter.textContent = '—'; lmFlicker.textContent = '—';
                }
            }
        };

        currentWs.onclose = () => {
            currentWs = null;
            videoScrubberContainer.style.display = 'none';
        };
    }


    // ── Select & Render ──────────────────────────────────────────────────────
    function selectFile(id) {
        // Close any existing stream
        if (currentWs) { currentWs.close(); currentWs = null; }

        currentFileId = id;
        document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
        const el = document.getElementById(`file-${id}`);
        if (el) el.classList.add('active');

        renderMainView();

        // Auto-start stream (paused) when a video is selected
        const fd = filesData.find(f => f.id === id);
        if (fd && fd.isVideo && fd.serverPath) {
            startVideoStream(true /* startPaused */);
        }
    }

    function renderMainView() {
        const fd = filesData.find(f => f.id === currentFileId);
        if (!fd) return;

        currentName.textContent = fd.file.name;

        if (fd.isVideo) {
            metricPolygons.style.display = 'none';
        }

        if (fd.processed && !fd.isVideo) {
            metricTime.style.display = 'inline-flex';
            metricTime.textContent = `⏱️ ${fd.timeMs}ms`;
            metricPolygons.style.display = 'inline-flex';
            metricPolygons.textContent = `🟦 ${fd.polygons} об'єктів`;
            metricFps.style.display = 'none';
        } else if (!fd.isVideo) {
            metricTime.style.display = metricPolygons.style.display = metricFps.style.display = 'none';
        }

        // Show placeholder only when images are genuinely absent from memory
        // (metrics-only upload) — toggle state doesn't affect already-processed files
        const hasNoImages = !fd.isVideo && fd.processed && !fd.originalB64 && !fd.segmentedB64;
        if (hasNoImages) {
            metricsOnlyPlaceholder.style.display = 'flex';
            wrapper.style.display = 'none';
            wrapperSide.style.display = 'none';
            return;
        }
        metricsOnlyPlaceholder.style.display = 'none';

        const gtToggleGroup = document.getElementById('gtToggleGroup');
        if (!fd.isVideo && fd.processed && fd.gtB64) {
            gtToggleGroup.style.display = 'inline-flex';
        } else {
            gtToggleGroup.style.display = 'none';
        }

        let segImg = fd.originalB64;
        if (!fd.isVideo && fd.processed) {
            if (currentGtView === 'gt' && fd.gtB64) {
                segImg = fd.gtB64;
            } else {
                if (currentFilter === 'masks') segImg = fd.segmentedMasksB64;
                else if (currentFilter === 'boxes') segImg = fd.segmentedBoxesB64;
                else segImg = fd.segmentedB64;
            }
        }

        if (currentViewMode === 'side') {
            wrapper.style.display = 'none';
            wrapperSide.style.display = 'flex';
            if (!fd.isVideo) { imgOrigSide.src = fd.originalB64 || ''; imgSegSide.src = segImg || ''; }
        } else {
            wrapper.style.display = 'block';
            wrapperSide.style.display = 'none';
            if (!fd.isVideo) imgSingle.src = (currentViewMode === 'original') ? (fd.originalB64 || '') : (segImg || '');
        }
    }

    // ── Pan & Zoom ───────────────────────────────────────────────────────────
    let scale = 1, panX = 0, panY = 0, isPanning = false, startX = 0, startY = 0;

    function activeWrapper() { return currentViewMode === 'side' ? wrapperSide : wrapper; }
    function applyTransform() {
        const t = activeWrapper();
        if (t) t.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    imgArea.addEventListener('mousedown', (e) => {
        if (e.target.tagName.toLowerCase() === 'input') return;
        if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
            isPanning = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            imgArea.style.cursor = 'grabbing';
            e.preventDefault();
        }
    });

    window.addEventListener('mouseup', () => {
        if (isPanning) { isPanning = false; imgArea.style.cursor = 'crosshair'; }
    });

    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        applyTransform();
    });

    imgArea.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 1 : -1;
        const rect = imgArea.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const oldSc = scale;
        scale = Math.min(Math.max(0.1, scale * Math.exp(delta * 0.1)), 10);
        panX = mx - (mx - panX) * (scale / oldSc);
        panY = my - (my - panY) * (scale / oldSc);
        applyTransform();
    }, { passive: false });
});
