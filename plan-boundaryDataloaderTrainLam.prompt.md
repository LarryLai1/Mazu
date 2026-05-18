## Plan: Boundary dataloader in train_LAM

目標是把 boundary-condition 的資料讀取路徑加進 [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py)，但先只做到資料集與 DataLoader 建立，不改訓練迴圈。這次已確認兩個關鍵前提：boundary 檔案以起報時間命名，且你希望 boundary loader 先獨立返回，不直接併進現有 regional batch。

**Steps**
1. 在 [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py) 新增 boundary 相關參數，至少包含 `boundary_root_dir` 與 halo 寬度控制參數，並讓它們保持可選，避免影響既有訓練指令。
2. 在 [datasets/]( /tmp3/b12902101/Mazu/datasets/ ) 底下新增一個 boundary dataset，專責讀取 forecast boundary condition 檔案、根據時間解析對應檔案、並切出指定 halo 區域。
3. boundary dataset 的格式與原先使用的 ERA5TWDatasetforAurora 一致，僅預報的資料是另外一個維度，叫做prediction_timedelta，數值為[0, 6, 12]。
4. boundary dataset的檔案結構類似於 data_root_dir，但是沒有static.nc，先查看該檔案結構後再決定具體的讀取邏輯，確保能根據 forecast 起報時間找到對應的 boundary 檔案。
5. 在 boundary dataset 內加入 mean pooling，將 0.25 度 boundary 資料先降到 0.5 度，再回傳給 DataLoader。
6. 修改 [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py) 的 `create_dataset()`，讓 train 和 val split 都能在需要時建立獨立的 boundary dataset，回傳方式維持和現有 regional dataset 分開。
7. 先不碰 `train_epoch()` 和 `val_epoch()`，維持這次改動只限於 loader 建立與資料輸出。
8. 補一個最小 smoke check，確認單一 sample 的時間對齊、halo 範圍、以及 pooling 後的 shape 都符合預期。

**Relevant files**
- [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py) — 加參數與 boundary loader 建立。
- [datasets/ERA5TWDatasetforAurora.py](/tmp3/b12902101/Mazu/datasets/ERA5TWDatasetforAurora.py) — 作為既有 regional dataset contract 參考。
- [datasets/]( /tmp3/b12902101/Mazu/datasets/ ) — 新增 boundary dataset 實作。
- [command.md](/tmp3/b12902101/Mazu/command.md) — boundary timing、halo 與解析度需求來源。
- [public_bash_scripts/train_LAM.sh](/tmp3/b12902101/Mazu/public_bash_scripts/train_LAM.sh) — 用來對照執行時參數傳遞。

**Verification**
1. 先確認 [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py) 仍可解析原本訓練參數，同時接受新的 boundary 參數。
2. 對新的 boundary dataset 做單筆 sample 驗證，檢查時間對齊、halo 切片與 mean pooling 後的 tensor shape。
3. 若 boundary loader 已接進 [train_LAM.py](/tmp3/b12902101/Mazu/train_LAM.py)，再做一次 focused import 或 syntax check。

**Decisions**
- boundary loading 這一版先獨立於現有 regional dataset。
- boundary 檔案命名先假設是以 forecast 起報時間為主。
- 現有 regional ERA5 的資料 contract 與訓練流程保持 backward-compatible。