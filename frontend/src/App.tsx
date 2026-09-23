// src/App.tsx

import React, { useEffect, useState } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";
import {
    Waves,
    TrendingUp,
    TrendingDown,
    Minus,
    AlertTriangle,
    RefreshCw,
} from "lucide-react";
import type { PredictionResponse, ChartPoint } from "./types";
import "./App.css";

const API_ENDPOINT = "http://127.0.0.1:8000/api/predict";

export default function App(): React.JSX.Element {
    const [data, setData] = useState<PredictionResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch prediction data from the API
    const fetchPrediction = async (): Promise<void> => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(API_ENDPOINT);
            if (!response.ok) {
                throw new Error(`Server status: ${response.status}`);
            }
            const result: PredictionResponse = await response.json();
            setData(result);
        } catch (err: unknown) {
            if (err instanceof Error) {
                setError(err.message);
            } else {
                setError("Failed to fetch model telemetry.");
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPrediction();
    }, []);

    // Render the delta indicator based on the delta value (positive, negative, or neutral)
    const renderDeltaIndicator = (delta: number): React.JSX.Element => {
        if (delta > 0.005) {
            return (
                <span className="metric-delta delta-positive">
                    <TrendingUp size={16} /> +{delta.toFixed(3)} m 
                </span>
            );
        } else if (delta < -0.005) {
            return (
                <span className="metric-delta delta-negative">
                    <TrendingDown size={16} /> {delta.toFixed(3)} m
                </span>
            );
        }
        return (
            <span className="metric-delta delta-neutral">
                <Minus size={16} /> 0.000 m
            </span>
        );
    };

    // Prepare chart points for the hydrograph trajectory chart
    const chartPoints: ChartPoint[] = data
        ? [
            {
                label: "Current",
                stage: data.current_stage_m,
                isForecast: false,
            },
            { label: "+1 Hour", stage: data.forecast_1h_m, isForecast: true },
            {
                label: "+3 Hours",
                stage: data.forecast_3h_m,
                isForecast: true,
            },
        ]
        : [];

    return (
        <div className="dashboard-container">
            {/* Header section with title and refresh button */}
            <header className="header">
                <div className="title-group">
                    <h1>Dartmoor Hydro-Surge</h1>
                    <p className="subtitle">
                        River Dart / Austin's Bridge Real-time Machine Learning
                        Telemetry
                    </p>
                </div>
                <button
                    className="card"
                    onClick={fetchPrediction}
                    disabled={loading}
                    style={{ cursor: "pointer" }}
                >
                    <RefreshCw size={16} className={loading ? "spin" : ""} />
                </button>
            </header>

            {/* Error banner if there's an error fetching data */}
            {error && (
                <div className="error-banner">
                    <AlertTriangle size={18} /> Error loading prediction:{" "}
                    {error}
                </div>
            )}

            {/* If data loads, display the metrics and chart */}
            {data && (
                <>
                    {/* Metrics grid displaying current river stage and forecasts */}
                    <div className="metrics-grid">
                        {/* Current River Stage Card */}
                        <div className="card">
                            <div className="card-title">
                                <Waves size={16} /> Current River Stage
                            </div>
                            <div className="metric-value">{data.current_stage_m.toFixed(3)} m</div>
                            <span className="subtitle">
                                Timestamp: {
                                (() => {
                                    // Ensure the string terminates with a 'Z' to force JS to parse it as UTC
                                    const utcTimestamp = data.timestamp.endsWith('Z') || data.timestamp.includes('+')
                                    ? data.timestamp 
                                    : `${data.timestamp}Z`;

                                    return new Date(utcTimestamp).toLocaleTimeString('en-GB', {
                                        timeZone: 'Europe/London',
                                        hour: '2-digit',
                                        minute: '2-digit',
                                        second: '2-digit'
                                    });
                                })()
                                }
                            </span>
                        </div>

                        {/* 1-Hour Forecast Card */}
                        <div className="card">
                            <div className="card-title">1-Hour Forecast</div>
                            <div className="metric-value">
                                {data.forecast_1h_m.toFixed(3)} m
                            </div>
                            {renderDeltaIndicator(data.delta_1h_m)}
                        </div>

                        {/* 3-Hour Forecast Card */}
                        <div className="card">
                            <div className="card-title">3-Hour Forecast</div>
                            <div className="metric-value">
                                {data.forecast_3h_m.toFixed(3)} m
                            </div>
                            {renderDeltaIndicator(data.delta_3h_m)}
                        </div>
                    </div>

                    {/* Hydrograph Trajectory Chart Card */}
                    <div className="chart-card">
                        <div className="chart-header">
                            <h2 className="chart-title">
                                Hydrograph Trajectory
                            </h2>
                            <span className="status-badge badge-normal">
                                XGBoost Inference Online
                            </span>
                        </div>

                        <div style={{ width: "100%", height: 320 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart
                                    data={chartPoints}
                                    margin={{
                                        top: 10,
                                        right: 30,
                                        left: 0,
                                        bottom: 0,
                                    }}
                                >
                                    <CartesianGrid
                                        strokeDasharray="3 3"
                                        stroke="#334155"
                                    />
                                    <XAxis dataKey="label" stroke="#94a3b8" />
                                    <YAxis
                                        stroke="#94a3b8"
                                        domain={[
                                            "dataMin - 0.05",
                                            "dataMax + 0.05",
                                        ]}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "#1e293b",
                                            borderColor: "#334155",
                                            color: "#f8fafc",
                                        }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="stage"
                                        stroke="#38bdf8"
                                        strokeWidth={3}
                                        dot={{ r: 6, fill: "#38bdf8" }}
                                    />
                                    <ReferenceLine
                                        y={2.0}
                                        label="Typical Surge Threshold"
                                        stroke="#f87171"
                                        strokeDasharray="3 3"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
