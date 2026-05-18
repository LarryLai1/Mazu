1. 邊界資料讀取規則：
    *   修改資料讀取邏輯 (Data Loader)：在預測區域模型時間點 t 時，提取全球模型中對應於 t-6 hr（過去）、t（現在）、t+6 hr（未來）這三個連續時間點的邊界資料。
    *   現在 (t) 與未來 (t+6) 必須讀取預測場（Forecast）：即使當下時間點有 Analysis，亦不得使用。必須沿用該預測循環或前一循環所產生的預測時效資料（例如：t=0 的現在讀取 f_-12+12；t=12 的現在讀取 f_0+12）。
    *   過去 (t-6) 可讀取預測場（Forecast）或分析場（Analysis）：若該時間點剛好有可用的分析場，允許直接讀取 Analysis（例如：t=6 的過去讀取 f_0+0；t=18 的過去讀取 f_12+0）。
    *   時間序列堆疊：將這三個時間點（過去、現在、未來）所讀取到的邊界資料依時間順序堆疊（Stack），作為時間序列特徵輸入。
2. 實作時間差編碼 (Time-difference Encoding)
    - 建立一個新的特徵通道 (Feature channel)，用於記錄每筆邊界資料「距離上一次分析 (Analysis) 初始化的時間差」。
    - 數值邏輯：這個數值可能是0(analysis), 6, 12(forecast)。
    - 對這些數值套用 正弦/餘弦編碼 (Sinusoidal positional encodings) 以產生時間差特徵張量。
3. 轉換boundary condition
    - 將boundary condition通過aurora preceiver轉換到boundary latent space，再通過一層MLP轉換到interior latent space的dimension
    - 將轉換後的三個boundary condition concat到interior data的latent dimension上