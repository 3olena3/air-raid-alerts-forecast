
import gradio as gr
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ==========================================
# 1. ГЛОБАЛЬНІ НАЛАШТУВАННЯ ТА URL
# ==========================================
GITHUB_RAW_URL = "https://raw.githubusercontent.com/username/repository/main/air_alerts.csv"


# ==========================================
# 2. ФУНКЦІЯ ПІДГОТОВКИ ДАНИХ ТА FEATURE ENGINEERING
# ==========================================
def prepare_feature_dataset(url):
    """
    Завантажує CSV та створює неперервну сітку з лаговими ознаками.
    """
    df = pd.read_csv(url)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['finished_at'] = pd.to_datetime(df['finished_at'])
    df = df.sort_values(by='started_at', ascending=True).reset_index(drop=True)
    
    df['duration_minutes'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0
    df['date'] = df['started_at'].dt.date
    
    # Агрегація за датою та областю
    daily_df = df.groupby(['date', 'oblast']).agg(
        alerts_count=('started_at', 'count'),
        total_duration_minutes=('duration_minutes', 'sum')
    ).reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    # Повна сітка (усі дати × усі області)
    all_dates = pd.date_range(start=daily_df['date'].min(), end=daily_df['date'].max(), freq='D')
    all_oblasts = daily_df['oblast'].unique()
    
    full_grid = pd.MultiIndex.from_product([all_dates, all_oblasts], names=['date', 'oblast']).to_frame().reset_index(drop=True)
    
    feature_df = pd.merge(full_grid, daily_df, on=['date', 'oblast'], how='left')
    feature_df['alerts_count'] = feature_df['alerts_count'].fillna(0).astype(int)
    feature_df['total_duration_minutes'] = feature_df['total_duration_minutes'].fillna(0.0).round(2)
    feature_df = feature_df.sort_values(by=['oblast', 'date']).reset_index(drop=True)
    
    # Генерація лагів та ковзних середніх
    for lag in [1, 2, 3, 7]:
        feature_df[f'alerts_count_lag_{lag}'] = feature_df.groupby('oblast')['alerts_count'].shift(lag)
        feature_df[f'duration_lag_{lag}'] = feature_df.groupby('oblast')['total_duration_minutes'].shift(lag)
        
    feature_df['alerts_count_roll_mean_7d'] = feature_df.groupby('oblast')['alerts_count'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    feature_df['duration_roll_mean_7d'] = feature_df.groupby('oblast')['total_duration_minutes'].transform(lambda x: x.shift(1).rolling(window=7).mean())
    
    feature_df['day_of_week'] = feature_df['date'].dt.dayofweek
    feature_df['is_weekend'] = feature_df['day_of_week'].isin([5, 6]).astype(int)
    
    return feature_df, list(all_oblasts)


# ==========================================
# ЕТАП 1: ГЛОБАЛЬНЕ НАВЧАННЯ МОДЕЛЕЙ
# ==========================================
print("🚀 Ініціалізація: Завантаження даних та навчання глобальних моделей...")

# 1. Завантаження та формування фічей
feature_df, all_oblasts = prepare_feature_dataset(GITHUB_RAW_URL)

# 2. Визначення дат
max_date = feature_df['date'].max()
valid_next_date = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

# 3. Визначення списку ознак
drop_cols = ['date', 'alerts_count', 'total_duration_minutes']
features = [col for col in feature_df.columns if col not in drop_cols]
cat_features = ['oblast']

# 4. Навчальний датасет
train_df = feature_df.dropna().reset_index(drop=True)
X_train = train_df[features]
y_train_alerts = train_df['alerts_count']
y_train_duration = train_df['total_duration_minutes']

# 5. Тренування двох глобальних моделей CatBoost
global_model_alerts = CatBoostRegressor(iterations=500, learning_rate=0.04, depth=6, random_seed=42, verbose=0)
global_model_alerts.fit(X_train, y_train_alerts, cat_features=cat_features)

global_model_duration = CatBoostRegressor(iterations=500, learning_rate=0.04, depth=6, random_seed=42, verbose=0)
global_model_duration.fit(X_train, y_train_duration, cat_features=cat_features)

# 6. Попереднє формування ознак для завтрашнього дня (valid_next_date)
recent_df = feature_df[feature_df['date'] > (max_date - pd.Timedelta(days=10))].copy()

next_date_dt = pd.to_datetime(valid_next_date)
next_day_rows = [{'date': next_date_dt, 'oblast': obl, 'alerts_count': np.nan, 'total_duration_minutes': np.nan} for obl in all_oblasts]
combined_df = pd.concat([recent_df, pd.DataFrame(next_day_rows)], ignore_index=True)
combined_df = combined_df.sort_values(by=['oblast', 'date']).reset_index(drop=True)

for lag in [1, 2, 3, 7]:
    combined_df[f'alerts_count_lag_{lag}'] = combined_df.groupby('oblast')['alerts_count'].shift(lag)
    combined_df[f'duration_lag_{lag}'] = combined_df.groupby('oblast')['total_duration_minutes'].shift(lag)

combined_df['alerts_count_roll_mean_7d'] = combined_df.groupby('oblast')['alerts_count'].transform(lambda x: x.shift(1).rolling(window=7).mean())
combined_df['duration_roll_mean_7d'] = combined_df.groupby('oblast')['total_duration_minutes'].transform(lambda x: x.shift(1).rolling(window=7).mean())

combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
combined_df['is_weekend'] = combined_df['day_of_week'].isin([5, 6]).astype(int)

# Вектор ознак для всіх областей на завтрашній день
global_target_features_df = combined_df[combined_df['date'] == next_date_dt].copy()

print(f"✅ Навчання завершено! Останній день у базі: {max_date.strftime('%Y-%m-%d')}")
print(f"🎯 Єдина доступна дата для прогнозу: {valid_next_date}\n")


# ==========================================
# ЕТАП 2: ФУНКЦІЯ ІНФЕРЕНСУ ДЛЯ GRADIO
# ==========================================
def predict_air_alerts(selected_oblast, user_date_str):
    """
    Функція, що перевіряє дату та викликає вже навчені моделі.
    """
    user_date_str = user_date_str.strip()
    
    # 1. Перевірка правильності введеної дати
    if user_date_str != valid_next_date:
        return (
            f"⚠️ **Попередження:** Прогноз доступний лише на наступний день після актуальної бази дати.\n\n"
            f"📅 **Останній день у базі даних:** `{max_date.strftime('%Y-%m-%d')}`\n"
            f"🎯 **Єдина дозволена дата для прогнозу:** `{valid_next_date}`"
        )
    
    # 2. Отримання готового вектора ознак для обраної області
    X_predict = global_target_features_df[global_target_features_df['oblast'] == selected_oblast][features]
    
    if X_predict.empty:
        return f"❌ Область '{selected_oblast}' не знайдена в базі даних."

    # 3. Прогноз через глобальні моделі
    pred_alerts = max(0, int(round(global_model_alerts.predict(X_predict)[0])))
    pred_duration = max(0.0, float(global_model_duration.predict(X_predict)[0]))
    
    # Форматування часу
    hours = int(pred_duration // 60)
    mins = int(pred_duration % 60)
    time_str = f"{hours} год {mins} хв" if hours > 0 else f"{mins} хв"
    
    return (
        f"📊 **Прогноз повітряних тривог для: {selected_oblast}**\n"
        f"📅 **Дата:** `{valid_next_date}`\n\n"
        f"🚨 **Очікувана кількість тривог:** `{pred_alerts}`\n"
        f"⏱️ **Очікувана загальна тривалість:** `{pred_duration:.1f} хв` (~{time_str})"
    )


# ==========================================
# 3. СТВОРЕННЯ ІНТЕРФЕЙСУ GRADIO
# ==========================================
default_oblast = sorted(all_oblasts)[0] if all_oblasts else "Київська область"

with gr.Blocks(title="Прогнозування Повітряних Тривог") as demo:
    gr.Markdown("# 🇺🇦 Система прогнозування повітряних тривог в Україні")
    gr.Markdown("Інтерактивний модуль прогнозування на основі вже навчених моделей CatBoost.")
    
    with gr.Row():
        with gr.Column():
            oblast_dropdown = gr.Dropdown(
                choices=sorted(all_oblasts), 
                value=default_oblast, 
                label="Оберіть область"
            )
            date_input = gr.Textbox(
                value=valid_next_date, 
                label="Цільова дата прогнозу (YYYY-MM-DD)",
                placeholder="YYYY-MM-DD"
            )
            submit_btn = gr.Button("Отримати прогноз", variant="primary")
            
        with gr.Column():
            output_box = gr.Markdown(label="Результат прогнозу")

    submit_btn.click(
        fn=predict_air_alerts,
        inputs=[oblast_dropdown, date_input],
        outputs=output_box
    )

# Запуск Gradio
demo.launch(share=True, debug=True)
