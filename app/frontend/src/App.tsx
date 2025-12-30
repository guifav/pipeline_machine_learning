import { useState } from 'react';
import './App.css';

// Tipos
interface FlightInput {
  month: number;
  day: number;
  day_of_week: number;
  hour: number;
  distance: number;
  scheduled_time: number;
  airline_delay_rate: number;
  origin_delay_rate: number;
  dest_delay_rate: number;
  route_delay_rate: number;
  origin_latitude: number;
  origin_longitude: number;
  dest_latitude: number;
  dest_longitude: number;
  origin_airport: number;
  destination_airport: number;
  origin_state: number;
  dest_state: number;
  is_weekend: number;
  is_morning_rush: number;
  is_evening_rush: number;
  is_domestic: number;
  [key: string]: number;
}

interface PredictionResponse {
  will_delay: boolean;
  delay_probability: number;
  estimated_delay_minutes: number | null;
  confidence: string;
}

const API_URL = 'http://localhost:8000';

function App() {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState<Partial<FlightInput>>({
    month: 6,
    day: 15,
    day_of_week: 3,
    hour: 14,
    distance: 1500,
    scheduled_time: 180,
    airline_delay_rate: 0.15,
    origin_delay_rate: 0.18,
    dest_delay_rate: 0.20,
    route_delay_rate: 0.17,
    origin_latitude: 33.9425,
    origin_longitude: -118.4081,
    dest_latitude: 40.7769,
    dest_longitude: -73.8740,
    origin_airport: 150,
    destination_airport: 200,
    origin_state: 5,
    dest_state: 32,
    is_weekend: 0,
    is_morning_rush: 0,
    is_evening_rush: 1,
    is_domestic: 1,
  });

  const [selectedAirline, setSelectedAirline] = useState('DL');
  const [selectedPeriod, setSelectedPeriod] = useState('Tarde');
  const [selectedSeason, setSelectedSeason] = useState('Verao');

  const airlines = ['AS', 'B6', 'DL', 'EV', 'F9', 'HA', 'MQ', 'NK', 'OO', 'UA', 'US', 'VX', 'WN'];
  const periods = ['Manha', 'Tarde', 'Noite'];
  const seasons = ['Primavera', 'Verao', 'Outono'];

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const payload: any = { ...formData };

      airlines.forEach(airline => {
        payload[`airline_${airline}`] = airline === selectedAirline ? 1 : 0;
      });

      periods.forEach(period => {
        payload[`period_of_day_${period}`] = period === selectedPeriod ? 1 : 0;
      });

      seasons.forEach(season => {
        payload[`season_${season}`] = season === selectedSeason ? 1 : 0;
      });

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido');
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field: string, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: typeof value === 'string' ? parseFloat(value) : value
    }));
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Flight Delay Predictor</h1>
        <p>Previsão de Atrasos em Voos usando Machine Learning</p>
      </header>

      <main className="main-content">
        <div className="form-container">
          <h2>Dados do Voo</h2>

          <div className="form-grid">
            <div className="form-section">
              <h3>Informações Temporais</h3>
              <div className="input-group">
                <label>Mês (1-12):</label>
                <input type="number" min="1" max="12" value={formData.month} onChange={(e) => updateField('month', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Dia (1-31):</label>
                <input type="number" min="1" max="31" value={formData.day} onChange={(e) => updateField('day', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Dia da Semana (1-7):</label>
                <input type="number" min="1" max="7" value={formData.day_of_week} onChange={(e) => updateField('day_of_week', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Hora (0-23):</label>
                <input type="number" min="0" max="23" value={formData.hour} onChange={(e) => updateField('hour', e.target.value)} />
              </div>
            </div>

            <div className="form-section">
              <h3>Informações Operacionais</h3>
              <div className="input-group">
                <label>Distância (milhas):</label>
                <input type="number" value={formData.distance} onChange={(e) => updateField('distance', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Tempo Programado (min):</label>
                <input type="number" value={formData.scheduled_time} onChange={(e) => updateField('scheduled_time', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Companhia Aérea:</label>
                <select value={selectedAirline} onChange={(e) => setSelectedAirline(e.target.value)}>
                  {airlines.map(airline => (<option key={airline} value={airline}>{airline}</option>))}
                </select>
              </div>
            </div>

            <div className="form-section">
              <h3>Taxas Históricas</h3>
              <div className="input-group">
                <label>Taxa Atraso Companhia:</label>
                <input type="number" step="0.01" min="0" max="1" value={formData.airline_delay_rate} onChange={(e) => updateField('airline_delay_rate', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Taxa Atraso Origem:</label>
                <input type="number" step="0.01" min="0" max="1" value={formData.origin_delay_rate} onChange={(e) => updateField('origin_delay_rate', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Taxa Atraso Destino:</label>
                <input type="number" step="0.01" min="0" max="1" value={formData.dest_delay_rate} onChange={(e) => updateField('dest_delay_rate', e.target.value)} />
              </div>
              <div className="input-group">
                <label>Taxa Atraso Rota:</label>
                <input type="number" step="0.01" min="0" max="1" value={formData.route_delay_rate} onChange={(e) => updateField('route_delay_rate', e.target.value)} />
              </div>
            </div>

            <div className="form-section">
              <h3>Outras Informações</h3>
              <div className="input-group">
                <label>Período do Dia:</label>
                <select value={selectedPeriod} onChange={(e) => setSelectedPeriod(e.target.value)}>
                  {periods.map(period => (<option key={period} value={period}>{period}</option>))}
                </select>
              </div>
              <div className="input-group">
                <label>Estação:</label>
                <select value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)}>
                  {seasons.map(season => (<option key={season} value={season}>{season}</option>))}
                </select>
              </div>
              <div className="input-group checkbox-group">
                <label><input type="checkbox" checked={formData.is_weekend === 1} onChange={(e) => updateField('is_weekend', e.target.checked ? 1 : 0)} /> É fim de semana</label>
              </div>
              <div className="input-group checkbox-group">
                <label><input type="checkbox" checked={formData.is_domestic === 1} onChange={(e) => updateField('is_domestic', e.target.checked ? 1 : 0)} /> Voo doméstico</label>
              </div>
            </div>
          </div>

          <button className="predict-button" onClick={handlePredict} disabled={loading}>
            {loading ? 'Processando...' : 'Prever Atraso'}
          </button>
        </div>

        {error && (
          <div className="error-box">
            <h3>Erro</h3>
            <p>{error}</p>
          </div>
        )}

        {prediction && (
          <div className={`prediction-box ${prediction.will_delay ? 'delayed' : 'on-time'}`}>
            <h2>Resultado da Previsão</h2>

            <div className="prediction-result">
              <div className="result-item main-result">
                <span className="result-label">Status:</span>
                <span className="result-value">{prediction.will_delay ? 'ATRASADO' : 'NO HORÁRIO'}</span>
              </div>

              <div className="result-item">
                <span className="result-label">Probabilidade de Atraso:</span>
                <span className="result-value">{(prediction.delay_probability * 100).toFixed(1)}%</span>
              </div>

              {prediction.estimated_delay_minutes !== null && (
                <div className="result-item">
                  <span className="result-label">Tempo Estimado de Atraso:</span>
                  <span className="result-value">{prediction.estimated_delay_minutes.toFixed(0)} minutos</span>
                </div>
              )}

              <div className="result-item">
                <span className="result-label">Confiança:</span>
                <span className={`result-value confidence-${prediction.confidence.toLowerCase()}`}>{prediction.confidence}</span>
              </div>
            </div>

            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${prediction.delay_probability * 100}%` }} />
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by Machine Learning | ROC-AUC: 0.75 | MAE: 17.8 min</p>
      </footer>
    </div>
  );
}

export default App;
