# Dose Ledger（束流站剂量记录账本）

束流站剂量记录的成批落盘服务。零第三方依赖，仅需 Python 3.11 标准库；
以独立 WAL 帧保证原子成批写入、崩溃截尾恢复、损坏隔离和乐观并发抢占。

## API

### `POST /api/batches`

```json
{ "expected_seq": 4, "records": [120, 95, 88] }
```

- `expected_seq`：调用方所见的下一序号（正整数，首个批次为 `1`）。
- `records`：1–32 条整数剂量记录。
- `201`：整帧 `write + fsync` 完成后才应答，帧内记录获得从 `seq` 起的
  连续序号，返回 `{"status":"committed","seq":4,"count":3,"next_seq":7}`。
- `409 stale_sequence`：所见序号已不是最新（并发抢占失败），**不写任何
  字节、不占用任何序号**，响应体带 `current_seq` 供调用方重读。
- `400`：载荷不合法（非整数、空批、超过 32 条、`expected_seq` 非正整数）。
- `503 ledger_poisoned`：WAL 已损坏。

### `GET /api/records?cursor=<seq>&limit=<n>`

按序号升序返回 `seq > cursor` 的记录；`limit` 1–1000（默认 1000）。
响应含 `next_cursor`，以该值为新游标读取下一页；返回空页时即到末尾。

```json
{ "records": [{"seq": 4, "dose": 120}, ...], "next_cursor": 6 }
```

poisoned 时返回 503，绝不返回可能误导的数据。

### `GET /healthz`

正常 `200 {"status":"ok","next_seq":N}`；poisoned 时 `503`。

## WAL 帧格式

```
+----------+----------+----------+----------------+--------+
| magic 8B | frame_no | length   | canonical JSON | SHA256 |
| DSL1WAL\n|  8B u64  | 8B u64   |  紧凑无空白    |  32B   |
+----------+----------+----------+----------------+--------+
```

- 摘要覆盖 **magic + frame_no + length + payload**，尾标 32 字节。
- 载荷为规范 JSON（`separators=(",", ":")`、UTF-8、保留记录在批内顺序）。
- 每帧一次 `write`，随后 `flush + fsync`（并尽力 fsync 父目录）；应答
  发生在持久化之后，因此不存在"应答了但没了"的批次。

## 恢复与 poisoned 语义

启动时顺序扫描 WAL：

- **唯一允许自动修复的情况**：物理文件末端的不完整帧（帧头写了一半、
  载荷写了一半，或末帧缺尾标字节）——它只可能是崩溃中断的那次追加，
  扫描器将文件**物理截尾**到最后一个完整帧边界（fsync），然后从完整
  帧重建连续序号。
- **以下情况一律令账本进入 poisoned**：完整帧 SHA-256 校验失败、魔数
  错误、长度字段非法（0 或超大且其后仍有帧标记 → 中部结构损坏）、
  帧序号断号/重号、载荷不是 1–32 条整数。poisoned 后禁止追加、禁止
  返回任何记录、healthz 为 503，且不改动现场文件。

长度字段损坏的判定：若某帧声明的边界越过 EOF，但在其"缺失区间"内仍能
找到后续帧魔数（整数 JSON 载荷中不可能出现该魔数），说明是中部损坏而非
截尾，必须 poisoned，避免误删其后的已确认帧。

## 并发

进程内一把互斥锁 + `expected_seq` 比较：同一所见序号并发提交时恰好一个
成功（201），其余全部 409 且在触碰 WAL 之前就返回——败者零字节、不占序号。

## 运行

```bash
docker compose up -d --build ledger
# 自定义宿主端口：
LEDGER_HOST_PORT=18080 docker compose up -d ledger
curl -s localhost:8080/healthz
```

数据在命名卷 `ledger-data`（容器内 `/data/wal.bin`）。
也可脱离容器直接运行：`LEDGER_WAL_PATH=./wal.bin python3 -m app.server`。

## 验证（compose 的 verify 服务）

```bash
docker compose run --rm verify
```

单次执行，退出码即结论，依次完成：

1. **构建检查** — `compileall` 字节码编译全部源码/测试/脚本；
2. **代码测试** — 27 个 `unittest` 用例（帧编码、截尾恢复、损坏隔离、
   并发抢占、游标分页、HTTP 503）；
3. **HTTP 冒烟** — `tests/e2e_smoke.py` 驱动真实服务子进程证明：
   - 写入途中 `os._exit` 杀死写入进程 → 重启物理截去半帧、序号连续重建；
   - 翻转完整帧内字节 → 重启后 health/读/写全部 503（poisoned 隔离）；
   - 16 客户端同序号 HTTP 并发 → 恰 1 个 201、15 个 409，WAL 精确增长
     一个帧（败者不留字节、不占序号）；
   - 并对 compose 中运行中的 ledger 容器再做一次 HTTP 提交+回读冒烟。

本地（无 Docker）可直接运行：`sh scripts/verify.sh`（退出码为 0 即通过）。
