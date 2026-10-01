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
