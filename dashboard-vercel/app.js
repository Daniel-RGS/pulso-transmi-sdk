async function fetchLeaderboard() {
    try {
        const response = await fetch(`/api/leaderboard`);
        if (!response.ok) throw new Error("API error");
        const data = await response.json();
        
        const students = data.data;
        let myData = null;
        
        // Find Daniel
        for (let i = 0; i < students.length; i++) {
            if (students[i].display_name.includes("Daniel Santiago Rincon")) {
                myData = students[i];
                break;
            }
        }
        
        if (myData) {
            document.getElementById("display-name").textContent = myData.display_name;
            document.getElementById("val-rank").textContent = `#${myData.rank}`;
            
            // Format Accuracy (animate counter if possible, for now just innerHTML)
            document.getElementById("val-accuracy").textContent = `${myData.accuracy.toFixed(2)}%`;
            document.getElementById("val-wape").textContent = myData.raw_wape.toFixed(4);
            document.getElementById("val-coverage").textContent = `${(myData.coverage * 100).toFixed(1)}%`;
        }

        // Render Table (Top 5 + Daniel if not in top 5)
        const tbody = document.querySelector("#leaderboard-table tbody");
        tbody.innerHTML = "";
        
        const top5 = students.slice(0, 5);
        let danielInTop5 = false;
        
        top5.forEach((student, index) => {
            if (student.display_name.includes("Daniel Santiago Rincon")) danielInTop5 = true;
            tbody.appendChild(createRow(student, index + 1));
        });
        
        if (myData && !danielInTop5) {
            // Add separator
            const sep = document.createElement("tr");
            sep.innerHTML = `<td colspan="5" style="text-align: center; color: #94a3b8;">...</td>`;
            tbody.appendChild(sep);
            
            // Add Daniel
            tbody.appendChild(createRow(myData, myData.rank, true));
        }
        
    } catch (error) {
        console.error("Error fetching leaderboard:", error);
    }
}

function createRow(student, rank, isDaniel = false) {
    const tr = document.createElement("tr");
    
    // Classes for styling
    if (rank <= 3) tr.className = `top-${rank}`;
    if (isDaniel || student.display_name.includes("Daniel Santiago Rincon")) {
        tr.classList.add("daniel-row");
    }
    
    tr.innerHTML = `
        <td><span class="rank-badge">${rank}</span></td>
        <td>${student.display_name}</td>
        <td>${student.accuracy.toFixed(2)}%</td>
        <td>${student.raw_wape.toFixed(4)}</td>
        <td><span style="color: ${student.coverage > 0 ? '#10b981' : '#ef4444'}">
            ${student.coverage > 0 ? 'Activo' : 'Inactivo'}
        </span></td>
    `;
    return tr;
}

// Initial fetch
fetchLeaderboard();

// Refresh every 30 seconds
setInterval(fetchLeaderboard, 30000);
