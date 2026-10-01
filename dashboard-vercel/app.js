/* ═══════════════════════════════════════════════════════
   CUSTOM CURSOR
══════════════════════════════════════════════════════ */
const cursor = document.getElementById('custom-cursor');

document.addEventListener('mousemove', e => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top  = e.clientY + 'px';
});
document.addEventListener('mousedown', () => cursor.classList.add('clicked'));
document.addEventListener('mouseup',   () => cursor.classList.remove('clicked'));

/* ═══════════════════════════════════════════════════════
   INTRO → BUS → DASHBOARD
══════════════════════════════════════════════════════ */
function startExperience() {
    const btn      = document.getElementById('btn-iniciar');
    const busScene = document.getElementById('bus-scene');
    const intro    = document.getElementById('intro-screen');
    const dash     = document.getElementById('dashboard');

    // Ocultar botón y mostrar escena del bus
    btn.style.display = 'none';
    busScene.classList.remove('hidden');

    // Cuando el bus termina su animación (3.2s) → mostrar dashboard
    setTimeout(() => {
        intro.classList.add('fade-out');
        dash.classList.remove('hidden');
        dash.classList.add('show');

        // Habilitar scroll del body
        document.body.style.overflow = 'auto';

        // Cargar datos del leaderboard
        fetchLeaderboard();
        setInterval(fetchLeaderboard, 30000);

        // Arrancar el colado Easter egg
        scheduleColado();
    }, 3300);
}

/* ═══════════════════════════════════════════════════════
   LEADERBOARD
══════════════════════════════════════════════════════ */
async function fetchLeaderboard() {
    try {
        const response = await fetch('/api/leaderboard');
        if (!response.ok) throw new Error('API error');
        const data = await response.json();
        const students = data.data;

        // Buscar a Daniel
        let myData = students.find(s => s.display_name.includes('Daniel Santiago Rincon'));

        if (myData) {
            document.getElementById('display-name').textContent = myData.display_name;
            document.getElementById('val-rank').textContent     = `#${myData.rank}`;
            document.getElementById('val-accuracy').textContent = `${myData.accuracy.toFixed(2)}%`;
            document.getElementById('val-wape').textContent     = myData.raw_wape.toFixed(4);
            document.getElementById('val-coverage').textContent = `${(myData.coverage * 100).toFixed(1)}%`;
        }

        // Renderizar todos los estudiantes
        const tbody = document.querySelector('#leaderboard-table tbody');
        tbody.innerHTML = '';
        students.forEach((student, index) => {
            tbody.appendChild(createRow(student, index + 1));
        });

    } catch (error) {
        console.error('Error fetching leaderboard:', error);
    }
}

function createRow(student, rank) {
    const tr = document.createElement('tr');

    if (rank <= 3) tr.className = `top-${rank}`;
    if (student.display_name.includes('Daniel Santiago Rincon')) {
        tr.classList.add('daniel-row');
    }

    const isActive = student.coverage > 0;
    const statusColor = isActive ? '#cc0000' : '#555';
    const statusText  = isActive ? '● Activo'  : '○ Inactivo';

    tr.innerHTML = `
        <td><span class="rank-badge">${rank}</span></td>
        <td>${student.display_name}</td>
        <td><strong>${student.accuracy.toFixed(2)}%</strong></td>
        <td>${student.raw_wape.toFixed(4)}</td>
        <td style="color:${statusColor}; font-weight:700">${statusText}</td>
    `;
    return tr;
}

/* ═══════════════════════════════════════════════════════
   EL COLADO — Easter egg (aparece cada ~45 segundos)
══════════════════════════════════════════════════════ */
const FRASES_COLADO = [
    '¡Sin pagar! 😂',
    '¡Pasaje qué! 🏃',
    '¡Ciencias de Datos! 🧠',
    '¡Externado rules! 💚',
    '¡A mí no me cobran! 😎',
    '¡Modelo random forest! 🌲',
];

function lanzarColado() {
    const scene   = document.getElementById('colado-scene');
    const student = document.getElementById('colado-student');
    const bubble  = document.getElementById('colado-bubble');
    if (!scene || student.classList.contains('running')) return;

    // Frase aleatoria
    bubble.textContent = FRASES_COLADO[Math.floor(Math.random() * FRASES_COLADO.length)];

    // Reset posición del estudiante (por si ya corrió antes)
    student.classList.remove('running');
    void student.offsetWidth; // force reflow

    // Mostrar escena
    scene.classList.remove('hidden');

    // Arrancar la carrera
    student.classList.add('running');

    // Mostrar globo justo cuando llega al centro (~42% de 5s = ~2.1s)
    setTimeout(() => bubble.classList.add('visible'), 2100);
    // Ocultar globo cuando sale (~60% = ~3s)
    setTimeout(() => bubble.classList.remove('visible'), 3100);

    // Limpiar todo al final
    setTimeout(() => {
        scene.classList.add('hidden');
        student.classList.remove('running');
    }, 5200);
}

// Primera aparición 20s después de entrar al dashboard,
// luego cada 50 segundos (con algo de aleatoriedad para sorprender)
function scheduleColado() {
    const delay = 20000 + Math.random() * 10000; // 20-30s primera vez
    setTimeout(() => {
        lanzarColado();
        // repetir cada 45-60s
        setInterval(() => lanzarColado(), 45000 + Math.random() * 15000);
    }, delay);
}

