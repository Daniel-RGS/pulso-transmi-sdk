export default async function handler(req, res) {
    const API_URL = "https://pulso-transmi.72-60-245-2.sslip.io";
    const API_KEY = process.env.PULSO_API_KEY;

    if (!API_KEY) {
        return res.status(500).json({ error: "Falta configurar PULSO_API_KEY en Vercel" });
    }

    try {
        const response = await fetch(`${API_URL}/v1/leaderboard?window=rolling_24h`, {
            headers: {
                "Authorization": `Bearer ${API_KEY}`
            }
        });
        
        if (!response.ok) {
            const errText = await response.text();
            return res.status(response.status).json({ error: "Error de la API del profesor", details: errText });
        }
        
        const data = await response.json();
        return res.status(200).json(data);
    } catch (error) {
        return res.status(500).json({ error: "Error de red al contactar la API", details: error.message });
    }
}
