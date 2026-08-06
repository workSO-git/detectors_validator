document.addEventListener('DOMContentLoaded', () => {

    // ── DOM refs ─────────────────────────────────────────────────────────────
    const fileInput       = document.getElementById('file-input');
    const btnUpload       = document.getElementById('btn-upload');
    const fileList        = document.getElementById('file-list');

    const wrapper         = document.getElementById('wrapper');
    const wrapperSide     = document.getElementById('wrapper-side');
    const imgSingle       = document.getElementById('img-single');
    const imgOrigSide     = document.getElementById('img-orig-side');
    const imgSegSide      = document.getElementById('img-seg-side');

    const currentName     = document.getElementById('current-name');
    const metricTime      = document.getElementById('metric-time');
    const metricFps       = document.getElementById('metric-fps');
    const metricPolygons  = document.getElementById('metric-polygons');

    const inputLabelsDir  = document.getElementById('input-labels-dir');
    const btnPickLabelsYaml = document.getElementById('btn-pick-labels-yaml');
    const photoMetrics    = document.getElementById('photo-metrics-panel');
    const pmGtBadge       = document.getElementById('pm-gt-badge');
    const pmIou           = document.getElementById('pm-iou');
    const pmPrec          = document.getElementById('pm-precision');
    const pmRecall        = document.getElementById('pm-recall');
    const pmF1            = document.getElementById('pm-f1');
    const pmCounts        = document.getElementById('pm-counts');

    const videoScrubberContainer = document.getElementById('video-scrubber-container');
    const btnPlayPause    = document.getElementById('btn-play-pause');
    const videoTimeEl     = document.getElementById('video-time');
    const videoSlider     = document.getElementById('video-slider');

    const lmStatus        = document.getElementById('lm-status');
    const lmIouAvg        = document.getElementById('lm-iou-avg');
    const lmIouMin        = document.getElementById('lm-iou-min');
    const lmJitter        = document.getElementById('lm-jitter');
    const lmFlicker       = document.getElementById('lm-flicker');

    const progressContainer = document.getElementById('progress-container');
    const progressBar       = document.getElementById('progress-bar');
    const progressText      = document.getElementById('progress-text');
    const imgArea           = document.getElementById('img-area');
    const dropZone          = document.getElementById('drop-zone');

    // ── State ────────────────────────────────────────────────────────────────
    let filesData        = [];
    let currentFileId    = null;
    let currentFilter    = 'all';
    let currentViewMode  = 'side';   // default: side-by-side
    let currentWs        = null;
    let totalFrames      = 0;
    let isDraggingSlider = false;
    let isVideoPaused    = true;     // default: paused
    let isProcessing     = false;

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
    setPillActive('modeGroup',   currentViewMode);

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
        const res  = await fetch('/api/pick_folder');
        const data = await res.json();
        return data.path || '';
    }

    async function pickFile() {
        const res  = await fetch('/api/pick_file');
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
            if (videoWs && videoWs.readyState === WebSocket.OPEN) {
                videoWs.send(JSON.stringify({ command: 'set_labels_dir', path }));
            }
        } else if (fd.processed) {
            // Re-evaluate current photo
            await reprocessPhotoMetrics(fd);
            renderMainView();
        }
    }

    btnPickLabelsYaml.addEventListener('click', async () => onLabelsPathPicked(await pickFile()));
    inputLabelsDir.addEventListener('change', onLabelsPathChanged);

    // ── File Upload & Drag'n'Drop ────────────────────────────────────────────
    btnUpload.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    // Prevent browser from opening dropped files natively
    window.addEventListener('dragover',  (e) => e.preventDefault(), false);
    window.addEventListener('drop',      (e) => e.preventDefault(), false);

    // Visual drop zone
    document.addEventListener('dragover',  () => dropZone.classList.add('active'));
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
            const file  = files[i];
            const isVid = file.type.startsWith('video/');
            if (!file.type.startsWith('image/') && !isVid) continue;

            const id          = Date.now() + '_' + i;
            const originalB64 = isVid ? null : await readAsDataURL(file);

            filesData.push({
                id, file, isVideo: isVid,
                serverPath: null, originalB64,
                segmentedB64: null, segmentedMasksB64: null, segmentedBoxesB64: null,
                timeMs: 0, polygons: 0, processed: false,
                metrics: null   // filled after processing if GT available
            });
            addToSidebar(id, file.name, isVid);
        }
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
                    const res  = await fetch('/api/upload_video?filename=' + encodeURIComponent(filesData[i].file.name), {
                        method: 'POST', body: filesData[i].file,
                        headers: { 'Content-Type': filesData[i].file.type || 'application/octet-stream' }
                    });
                    const data = await res.json();
                    if (data.success) {
                        filesData[i].serverPath = data.path;
                        filesData[i].processed  = true;
                        markDone(filesData[i].id, '🎬');
                        if (currentFileId === filesData[i].id) renderMainView();
                    }
                } else {
                    const fd2 = new FormData();
                    fd2.append('file', filesData[i].file);
                    const labelsDir = inputLabelsDir.value.trim();
                    if (labelsDir) fd2.append('labels_dir', labelsDir);
                    const res  = await fetch('/api/process', { method: 'POST', body: fd2 });
                    const data = await res.json();
                    if (data.success) {
                        Object.assign(filesData[i], {
                            segmentedB64:      data.image,
                            segmentedMasksB64: data.image_masks,
                            segmentedBoxesB64: data.image_boxes,
                            timeMs:            data.time_ms,
                            polygons:          data.polygons,
                            processed:         true,
                            metrics:           data.metrics || null
                        });
                        markDone(filesData[i].id, '✅');
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
    }

    function markDone(id, icon) {
        const el = document.getElementById(`file-${id}`);
        if (!el) return;
        el.classList.add('processed');
        const s = el.querySelector('.file-ok');
        if (s) s.textContent = icon;
    }

    function updateProgress(done, total) {
        progressText.textContent = `${done} / ${total}`;
        progressBar.style.width  = `${(done / total) * 100}%`;
    }

    // ── Re-evaluate photo metrics with a new labels dir ──────────────────
    async function reprocessPhotoMetrics(fdEntry) {
        const labelsDir = inputLabelsDir.value.trim();
        if (!labelsDir || fdEntry.isVideo) return;
        try {
            const fd2 = new FormData();
            fd2.append('file', fdEntry.file);
            fd2.append('labels_dir', labelsDir);
            const res  = await fetch('/api/process', { method: 'POST', body: fd2 });
            const data = await res.json();
            if (data.success) {
                fdEntry.metrics = data.metrics || null;
            }
        } catch (e) { console.error(e); }
    }

    // ── Video scrubber ───────────────────────────────────────────────────────
    videoSlider.addEventListener('mousedown', () => { isDraggingSlider = true; });
    videoSlider.addEventListener('mouseup',   () => { isDraggingSlider = false; });
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
        lmStatus.className   = 'lm-status';
        lmIouAvg.textContent = '—';
        lmIouMin.textContent = '—';
        lmJitter.textContent  = '—';
        lmFlicker.textContent = '—';
    }

    function startVideoStream(startPaused) {
        const fd = filesData.find(f => f.id === currentFileId);
        if (!fd || !fd.isVideo || !fd.serverPath) return;
        if (currentWs) return; // already streaming

        isVideoPaused = startPaused !== false; // default: paused
        btnPlayPause.textContent = isVideoPaused ? '▶️' : '⏸️';

        const wsUrl = `ws://${window.location.host}/ws/video?path=${encodeURIComponent(fd.serverPath)}`;
        currentWs   = new WebSocket(wsUrl);

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
                currentWs.send(JSON.stringify({ command: 'set_filter', filter: currentFilter }));
                currentWs.send(JSON.stringify({ command: 'set_labels_dir', path: inputLabelsDir.value.trim() }));
                // Send pause command immediately if starting paused
                if (isVideoPaused) {
                    currentWs.send(JSON.stringify({ command: 'pause', state: true }));
                }
                return;
            }

            if (data.type === 'done') { currentWs.close(); return; }
            if (data.error) { alert(data.error); return; }

            if (data.type === 'frame') {
                if (currentViewMode === 'side') {
                    imgOrigSide.src = data.image_orig;
                    imgSegSide.src  = data.image;
                } else if (currentViewMode === 'original') {
                    imgSingle.src = data.image_orig;
                } else {
                    imgSingle.src = data.image;
                }

                metricFps.style.display  = 'inline-flex';
                metricFps.textContent    = `⚡ ${data.fps} FPS`;
                metricTime.style.display = 'inline-flex';
                metricTime.textContent   = `⏱️ ${data.time_ms}ms`;

                if (!isDraggingSlider) {
                    videoSlider.value       = data.frame_idx;
                    videoTimeEl.textContent = `${data.frame_idx} / ${totalFrames}`;
                }

                const m = data.metrics;
                if (!m) { resetLiveMetrics(); return; }

                if (m.warming_up) {
                    lmStatus.textContent = 'Warming Up…';
                    lmStatus.className   = 'lm-status warming';
                    lmIouAvg.textContent = '—'; lmIouMin.textContent = '—';
                    lmJitter.textContent = '—'; lmFlicker.textContent = '—';
                } else if (m.has_data) {
                    lmStatus.textContent = 'Running';
                    lmStatus.className   = 'lm-status running';
                    lmIouAvg.textContent = m.iou_avg.toFixed(3);
                    lmIouMin.textContent = m.iou_min.toFixed(3);
                    lmJitter.textContent  = m.jitter.toFixed(1);
                    lmFlicker.textContent = m.flicker.toFixed(1) + '%';
                } else {
                    lmStatus.textContent = 'Без GT-файлів';
                    lmStatus.className   = 'lm-status';
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
            photoMetrics.style.display   = 'none';
        }

        if (fd.processed && !fd.isVideo) {
            metricTime.style.display     = 'inline-flex';
            metricTime.textContent       = `⏱️ ${fd.timeMs}ms`;
            metricPolygons.style.display = 'inline-flex';
            metricPolygons.textContent   = `🟦 ${fd.polygons} об'єктів`;
            metricFps.style.display      = 'none';

            // Photo metrics panel
            const m = fd.metrics;
            if (m && m.has_gt) {
                photoMetrics.style.display = 'block';
                pmGtBadge.textContent      = '✅ з GT';
                pmGtBadge.className        = 'gt-badge gt-yes';
                pmIou.textContent          = m.iou.toFixed(3);
                pmPrec.textContent         = m.precision.toFixed(3);
                pmRecall.textContent       = m.recall.toFixed(3);
                pmF1.textContent           = m.f1.toFixed(3);
                pmCounts.textContent       = `Pred: ${m.objects}  |  GT: ${m.gt_count}`;
            } else if (m && !m.has_gt) {
                photoMetrics.style.display = 'block';
                pmGtBadge.textContent      = '❌ без GT';
                pmGtBadge.className        = 'gt-badge gt-no';
                pmIou.textContent = pmPrec.textContent = pmRecall.textContent = pmF1.textContent = '—';
                pmCounts.textContent = m.reason || 'Файл GT не знайдено';
            } else {
                photoMetrics.style.display = 'none';
            }
        } else if (!fd.isVideo) {
            metricTime.style.display = metricPolygons.style.display = metricFps.style.display = 'none';
            photoMetrics.style.display = 'none';
        }

        let segImg = fd.originalB64;
        if (!fd.isVideo && fd.processed) {
            if (currentFilter === 'masks')      segImg = fd.segmentedMasksB64;
            else if (currentFilter === 'boxes') segImg = fd.segmentedBoxesB64;
            else                                segImg = fd.segmentedB64;
        }

        if (currentViewMode === 'side') {
            wrapper.style.display     = 'none';
            wrapperSide.style.display = 'flex';
            if (!fd.isVideo) { imgOrigSide.src = fd.originalB64; imgSegSide.src = segImg; }
        } else {
            wrapper.style.display     = 'block';
            wrapperSide.style.display = 'none';
            if (!fd.isVideo) imgSingle.src = (currentViewMode === 'original') ? fd.originalB64 : segImg;
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
        const delta  = e.deltaY < 0 ? 1 : -1;
        const rect   = imgArea.getBoundingClientRect();
        const mx     = e.clientX - rect.left;
        const my     = e.clientY - rect.top;
        const oldSc  = scale;
        scale = Math.min(Math.max(0.1, scale * Math.exp(delta * 0.1)), 10);
        panX  = mx - (mx - panX) * (scale / oldSc);
        panY  = my - (my - panY) * (scale / oldSc);
        applyTransform();
    }, { passive: false });
});
