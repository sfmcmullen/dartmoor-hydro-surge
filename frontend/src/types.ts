// src/types.ts

export interface PredictionResponse {
    timestamp: string;
    current_stage_m: number;
    forecast_1h_m: number;
    forecast_3h_m: number;
    delta_1h_m: number;
    delta_3h_m: number;
}

export interface ChartPoint {
    label: string;
    stage: number;
    isForecast: boolean;
}