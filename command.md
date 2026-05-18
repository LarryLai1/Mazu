0. 建構讀取boundary condition資訊的dataset與dataloader
    - 全球模型資料規格：Forecast 每隔 12 小時有一筆新起報（如 00, 12, 24 hr...），每筆預測的輸出時間步長（timestep）為 6 小時，其中 0 hr 的資料即為該時間點的 Analysis data。
    - 通過變數控制要使用的boundary condition往外的格數(使用argparse，0.5度)
    - 先使用mean pooling將boundary condition從0.25度轉為0.5度，模擬低解析度的預報；再將其使用bilinear interpolation轉換回原始的0.25度