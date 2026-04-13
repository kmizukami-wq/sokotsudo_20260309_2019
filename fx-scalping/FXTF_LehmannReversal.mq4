//+------------------------------------------------------------------+
//| FXTF_LehmannReversal.mq4                                         |
//| Lehmann 高速逆転 (HFT mean-reversion) 戦略                       |
//| FX TF ランク1 ゼロスプレッド + 手数料0円 環境専用                |
//|                                                                  |
//| 数学的根拠:                                                      |
//|   Lehmann (1990), Lo & MacKinlay (1990) が実証した              |
//|   短期 (1秒〜数秒) リターンの負の自己相関 ρ(1) ≈ -0.05〜-0.15   |
//|                                                                  |
//| 戦略:                                                            |
//|   1. 過去 InpReversalLookbackMs (例:1000ms) 内に                 |
//|      |move| > InpReversalThresholdPip 動いたら逆方向にエントリ   |
//|   2. 退出:                                                       |
//|      - +InpTakeProfitPip 到達 (例: +0.5p)                       |
//|      - -InpStopLossPip 到達 (例: -0.7p)                         |
//|      - InpMaxHoldMs 経過 (例: 5000ms = 5秒)                     |
//|                                                                  |
//| 重要: ゼロスプレッド前提。通常スプレッドでは絶対に勝てない       |
//+------------------------------------------------------------------+
#property copyright   "FXTF LehmannReversal"
#property link        ""
#property version     "1.00"
#property strict
#property description "FXTF Rank1 zero-spread HFT mean-reversion (Lehmann 1990)"

//--- Input parameters
input double InpLots                  = 0.01;   // Lot size (0.01=100units / 0.10=1,000units)
input int    InpMagicNumber           = 0;      // Magic (0 = auto-generate from Symbol, base 86000)
input int    InpSlippage              = 0;      // Max slippage (points); 0 = strict

input string _sep1_                   = "===== Lehmann HFT Reversal =====";
input int    InpReversalLookbackMs    = 1000;   // Lookback window (msec) for move detection
input double InpReversalThresholdPip  = 0.5;    // Min move (pip) in lookback to trigger fade
input double InpTakeProfitPip         = 0.5;    // Take profit (pip)
input double InpStopLossPip           = 0.7;    // Stop loss (pip)
input int    InpMaxHoldMs             = 5000;   // Max hold (msec)
input int    InpMinIntervalMs         = 200;    // Min ms between consecutive entries (anti-spam)
input int    InpTickBufferSize        = 1000;   // Ring buffer size (ticks)

input string _sep2_                   = "===== Risk Guard =====";
input double InpMaxUnits              = 10000;  // Rank-1 unit cap (0-fee tier)
input double InpMaxSpreadPoints       = 2;      // Max allowed spread (points); strict for HFT
input bool   InpSkipTokyoMorn         = false;  // Skip JST 06-09
input bool   InpSkipNYNewsHrs         = false;  // Skip JST 21-24
input int    InpJSTFromBroker         = 6;      // Broker -> JST offset (summer=6 / winter=7)
input bool   InpTradeOnlyEAHrs        = true;   // Enable custom hours
input int    InpStartHour             = 9;      // Start hour JST
input int    InpEndHour               = 3;      // End hour JST (wraps)
input bool   InpVerbose               = false;  // Verbose log
input bool   InpLogCSV                = false;  // Append CSV to MQL4/Files/LehmannRev_<Symbol>.csv

//--- Globals: tick ring buffer with timestamps
double   g_tickMid[];        // mid price ring buffer
uint     g_tickMs[];         // GetTickCount() at each tick
int      g_tickHead   = 0;   // next write index
int      g_tickCount  = 0;   // valid items

double   g_point;
double   g_pipSize;
int      g_effectiveMagic = 0;

//--- Position state
uint     g_entryMs        = 0;   // GetTickCount() at entry
int      g_entrySide      = 0;
double   g_entryPrice     = 0;
double   g_entryMovePip   = 0;   // 検出時の move pip
int      g_entryLookbackMs = 0;
uint     g_lastEntryMs    = 0;   // 連続発火防止用

string   g_csvFile = "";

//+------------------------------------------------------------------+
//| Symbol からマジックナンバーを自動生成 (86000-94999 の範囲)         |
//+------------------------------------------------------------------+
int AutoMagicFromSymbol()
{
   string s = Symbol();
   int hash = 0;
   int len = StringLen(s);
   for(int i = 0; i < len; i++)
   {
      int c = StringGetChar(s, i);
      hash = (hash * 31 + c) & 0x7FFFFFFF;
   }
   return 86000 + (hash % 9000);  // 86000〜94999 (TickScalper 77000-85999 と分離)
}

//+------------------------------------------------------------------+
//| Side -> string                                                   |
//+------------------------------------------------------------------+
string SideStr(int s)
{
   if(s > 0) return "BUY";
   if(s < 0) return "SELL";
   return "FLAT";
}

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   ArrayResize(g_tickMid, InpTickBufferSize);
   ArrayResize(g_tickMs,  InpTickBufferSize);
   ArrayInitialize(g_tickMid, 0.0);
   ArrayInitialize(g_tickMs,  0);
   g_tickHead = 0;
   g_tickCount = 0;

   g_point = Point;
   if(Digits == 3 || Digits == 5)
      g_pipSize = g_point * 10;
   else
      g_pipSize = g_point;

   if(InpMagicNumber <= 0)
      g_effectiveMagic = AutoMagicFromSymbol();
   else
      g_effectiveMagic = InpMagicNumber;

   Print("=== FXTF LehmannReversal initialized ===");
   PrintFormat("Symbol=%s Digits=%d Point=%.5f pipSize=%.5f Magic=%d%s",
               Symbol(), Digits, g_point, g_pipSize,
               g_effectiveMagic, (InpMagicNumber<=0 ? " (auto)" : ""));

   datetime brokerT = TimeCurrent();
   int jstHour = (TimeHour(brokerT) + InpJSTFromBroker) % 24;
   int jstMin = TimeMinute(brokerT);
   PrintFormat("Time: broker=%s JST=%02d:%02d (offset=%d)",
               TimeToStr(brokerT, TIME_DATE|TIME_MINUTES),
               jstHour, jstMin, InpJSTFromBroker);
   PrintFormat("Trade hours (JST): %02d:00 to %02d:00 (TradeOnly=%s)",
               InpStartHour, InpEndHour,
               (InpTradeOnlyEAHrs ? "true" : "false"));
   PrintFormat("Strategy: lookback=%dms threshold=%.2fp TP=%.2fp SL=%.2fp maxHold=%dms",
               InpReversalLookbackMs, InpReversalThresholdPip,
               InpTakeProfitPip, InpStopLossPip, InpMaxHoldMs);
   PrintFormat("Execution: lots=%.2f slippage=%d maxSpread=%.1fp minInterval=%dms",
               InpLots, InpSlippage, InpMaxSpreadPoints, InpMinIntervalMs);
   PrintFormat("WARNING: Designed for ZERO spread only. Do NOT use on standard spread accounts.");

   g_csvFile = StringConcatenate("LehmannRev_", Symbol(), ".csv");
   if(InpLogCSV)
   {
      int h = FileOpen(g_csvFile, FILE_CSV|FILE_READ|FILE_ANSI, ';');
      bool needHeader = (h == INVALID_HANDLE);
      if(h != INVALID_HANDLE) FileClose(h);
      if(needHeader)
      {
         int hw = FileOpen(g_csvFile, FILE_CSV|FILE_WRITE|FILE_ANSI, ';');
         if(hw != INVALID_HANDLE)
         {
            FileWriteString(hw, "timestamp;symbol;event;side;pip;holdMs;movePip;lookbackMs;spread\r\n");
            FileClose(hw);
         }
      }
      PrintFormat("CSV log -> MQL4/Files/%s", g_csvFile);
   }
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {}

//+------------------------------------------------------------------+
//| CSV writer                                                       |
//+------------------------------------------------------------------+
void LogCSV(string line)
{
   if(!InpLogCSV) return;
   int h = FileOpen(g_csvFile, FILE_CSV|FILE_READ|FILE_WRITE|FILE_ANSI, ';');
   if(h == INVALID_HANDLE)
   {
      h = FileOpen(g_csvFile, FILE_CSV|FILE_WRITE|FILE_ANSI, ';');
      if(h == INVALID_HANDLE) return;
   }
   FileSeek(h, 0, SEEK_END);
   FileWriteString(h, line + "\r\n");
   FileClose(h);
}

//+------------------------------------------------------------------+
//| JST hour                                                         |
//+------------------------------------------------------------------+
int GetJSTHour()
{
   int h = TimeHour(TimeCurrent()) + InpJSTFromBroker;
   h = (h % 24 + 24) % 24;
   return h;
}

//+------------------------------------------------------------------+
//| Trade hour filter                                                |
//+------------------------------------------------------------------+
bool IsTradeHour()
{
   int jstH = GetJSTHour();
   if(InpSkipTokyoMorn && jstH >= 6 && jstH < 9) return false;
   if(InpSkipNYNewsHrs && jstH >= 21 && jstH < 24) return false;
   if(InpTradeOnlyEAHrs)
   {
      if(InpStartHour <= InpEndHour)
         return (jstH >= InpStartHour && jstH < InpEndHour);
      return (jstH >= InpStartHour || jstH < InpEndHour);
   }
   return true;
}

//+------------------------------------------------------------------+
//| Spread check                                                     |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable()
{
   double spreadPoints = (Ask - Bid) / g_point;
   if(spreadPoints > InpMaxSpreadPoints)
   {
      if(InpVerbose)
         PrintFormat("Spread %.1fp > max %.1fp, skip", spreadPoints, InpMaxSpreadPoints);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Push tick to ring buffer                                         |
//+------------------------------------------------------------------+
void PushTick(double mid, uint nowMs)
{
   g_tickMid[g_tickHead] = mid;
   g_tickMs[g_tickHead]  = nowMs;
   g_tickHead = (g_tickHead + 1) % InpTickBufferSize;
   if(g_tickCount < InpTickBufferSize) g_tickCount++;
}

//+------------------------------------------------------------------+
//| Find mid price at approximately (nowMs - targetAgoMs)            |
//| Returns 0 if not enough buffer history                           |
//+------------------------------------------------------------------+
double GetMidAtMsAgo(uint nowMs, int targetAgoMs)
{
   if(g_tickCount < 2) return 0;
   uint targetMs = nowMs - (uint)targetAgoMs;
   // search backwards from head; find first tick with ms <= targetMs
   for(int i = 1; i <= g_tickCount; i++)
   {
      int idx = (g_tickHead - i + InpTickBufferSize) % InpTickBufferSize;
      if(g_tickMs[idx] <= targetMs)
         return g_tickMid[idx];
   }
   // older than buffer; use oldest
   int oldest = (g_tickHead - g_tickCount + InpTickBufferSize) % InpTickBufferSize;
   return g_tickMid[oldest];
}

//+------------------------------------------------------------------+
//| Compute move (pip) over last InpReversalLookbackMs               |
//+------------------------------------------------------------------+
double ComputeRecentMovePip(uint nowMs, double currentMid)
{
   double pastMid = GetMidAtMsAgo(nowMs, InpReversalLookbackMs);
   if(pastMid <= 0) return 0;
   return (currentMid - pastMid) / g_pipSize;
}

//+------------------------------------------------------------------+
//| Find my open position (-1 if none)                               |
//+------------------------------------------------------------------+
int FindMyPosition()
{
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != Symbol()) continue;
      if(OrderMagicNumber() != g_effectiveMagic) continue;
      if(OrderType() != OP_BUY && OrderType() != OP_SELL) continue;
      return OrderTicket();
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Sum same-direction units across all orders on this symbol        |
//+------------------------------------------------------------------+
double SumSameDirectionUnits(int side)
{
   double total = 0;
   double contractSize = MarketInfo(Symbol(), MODE_LOTSIZE);
   for(int i = OrdersTotal()-1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != Symbol()) continue;
      int t = OrderType();
      bool isBuy  = (t == OP_BUY  || t == OP_BUYLIMIT  || t == OP_BUYSTOP);
      bool isSell = (t == OP_SELL || t == OP_SELLLIMIT || t == OP_SELLSTOP);
      if(side > 0 && isBuy)  total += OrderLots() * contractSize;
      if(side < 0 && isSell) total += OrderLots() * contractSize;
   }
   return total;
}

//+------------------------------------------------------------------+
//| Open position                                                    |
//+------------------------------------------------------------------+
void OpenPosition(int side, double movePip, uint nowMs)
{
   double contractSize = MarketInfo(Symbol(), MODE_LOTSIZE);
   double newUnits = InpLots * contractSize;
   double existing = SumSameDirectionUnits(side);
   if(existing + newUnits > InpMaxUnits)
   {
      if(InpVerbose)
         PrintFormat("Guard: existing=%.0f + new=%.0f > max=%.0f, skip",
                     existing, newUnits, InpMaxUnits);
      return;
   }

   int type = (side > 0) ? OP_BUY : OP_SELL;
   double price = (side > 0) ? Ask : Bid;
   string comment = "Lehmann";
   int ticket = OrderSend(Symbol(), type, InpLots, price, InpSlippage,
                          0, 0, comment, g_effectiveMagic, 0, clrNONE);
   if(ticket < 0)
   {
      int err = GetLastError();
      string hint = "";
      switch(err)
      {
         case 129: hint = "invalid price - stale quote"; break;
         case 131: hint = "invalid lot step"; break;
         case 132: hint = "market closed"; break;
         case 133: hint = "trade disabled"; break;
         case 134: hint = "not enough money"; break;
         case 135: hint = "price changed - retry next tick"; break;
         case 136: hint = "off quotes"; break;
         case 138: hint = "requote - raise InpSlippage if frequent"; break;
         case 146: hint = "trade context busy"; break;
         case 149: hint = "hedging prohibited"; break;
         default:  hint = "see MT4 error code reference"; break;
      }
      PrintFormat("OrderSend failed err=%d (%s)", err, hint);
      return;
   }

   g_entryMs       = nowMs;
   g_entrySide     = side;
   g_entryPrice    = price;
   g_entryMovePip  = movePip;
   g_entryLookbackMs = InpReversalLookbackMs;
   g_lastEntryMs   = nowMs;

   double spread = (Ask - Bid) / g_point;
   PrintFormat("ENTRY %s @%.5f ticket=%d movePip=%+.2fp lookback=%dms spread=%.1fp",
               SideStr(side), price, ticket, movePip,
               InpReversalLookbackMs, spread);

   if(InpLogCSV)
   {
      string line = StringFormat("%s;%s;ENTRY;%s;;;%+0.2f;%d;%.1f",
                                 TimeToStr(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                                 Symbol(), SideStr(side),
                                 movePip, InpReversalLookbackMs, spread);
      LogCSV(line);
   }
}

//+------------------------------------------------------------------+
//| Current pip P/L                                                  |
//+------------------------------------------------------------------+
double CurrentPipPnL(int ticket)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return 0;
   double entry = OrderOpenPrice();
   double cur   = (OrderType() == OP_BUY) ? Bid : Ask;
   int side     = (OrderType() == OP_BUY) ? 1 : -1;
   return (cur - entry) * side / g_pipSize;
}

//+------------------------------------------------------------------+
//| Close position                                                   |
//+------------------------------------------------------------------+
void ClosePosition(int ticket, string reason, uint nowMs)
{
   if(!OrderSelect(ticket, SELECT_BY_TICKET)) return;
   double price = (OrderType() == OP_BUY) ? Bid : Ask;
   double pip   = CurrentPipPnL(ticket);
   int side     = (OrderType() == OP_BUY) ? 1 : -1;
   int holdMs   = (int)(nowMs - g_entryMs);

   bool ok = OrderClose(ticket, OrderLots(), price, InpSlippage, clrNONE);
   if(!ok)
   {
      PrintFormat("OrderClose failed err=%d", GetLastError());
      return;
   }
   PrintFormat("EXIT[%s] side=%s pip=%+.2f holdMs=%d ticket=%d (entry move=%+.2fp lookback=%dms)",
               reason, SideStr(side), pip, holdMs, ticket,
               g_entryMovePip, g_entryLookbackMs);

   if(InpLogCSV)
   {
      string evt = "EXIT_" + reason;
      double spread = (Ask - Bid) / g_point;
      string line = StringFormat("%s;%s;%s;%s;%+0.2f;%d;%+0.2f;%d;%.1f",
                                 TimeToStr(TimeCurrent(), TIME_DATE|TIME_SECONDS),
                                 Symbol(), evt, SideStr(side),
                                 pip, holdMs,
                                 g_entryMovePip, g_entryLookbackMs, spread);
      LogCSV(line);
   }
}

//+------------------------------------------------------------------+
//| Manage open position (TP / SL / timeout)                         |
//+------------------------------------------------------------------+
void ManagePosition(int ticket, uint nowMs)
{
   double pip = CurrentPipPnL(ticket);
   int holdMs = (int)(nowMs - g_entryMs);

   // 利確
   if(pip >= InpTakeProfitPip)
   {
      ClosePosition(ticket, "TP", nowMs);
      return;
   }
   // 損切り
   if(pip <= -InpStopLossPip)
   {
      ClosePosition(ticket, "SL", nowMs);
      return;
   }
   // タイムアウト
   if(holdMs >= InpMaxHoldMs)
   {
      ClosePosition(ticket, "TIMEOUT", nowMs);
      return;
   }
}

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   uint nowMs = GetTickCount();
   double mid = (Bid + Ask) / 2.0;
   PushTick(mid, nowMs);

   // 既存ポジション管理
   int ticket = FindMyPosition();
   if(ticket >= 0)
   {
      ManagePosition(ticket, nowMs);
      return;
   }

   // ゲート
   if(!IsTradeHour()) return;
   if(!IsSpreadAcceptable()) return;

   // 連続発火防止
   if(g_lastEntryMs > 0 && (nowMs - g_lastEntryMs) < (uint)InpMinIntervalMs)
      return;

   // バッファ充填チェック (lookback分の履歴が必要)
   if(g_tickCount < 5) return;

   // Lehmann シグナル評価
   double movePip = ComputeRecentMovePip(nowMs, mid);
   if(movePip == 0) return;

   if(movePip > InpReversalThresholdPip)
   {
      // 急騰直後 → SELL (Lehmann reversal)
      OpenPosition(-1, movePip, nowMs);
   }
   else if(movePip < -InpReversalThresholdPip)
   {
      // 急落直後 → BUY
      OpenPosition(+1, movePip, nowMs);
   }
   else if(InpVerbose)
   {
      static uint lastVerbose = 0;
      if(nowMs - lastVerbose > 5000)  // 5秒に1回だけ
      {
         PrintFormat("[gate] movePip=%+.2fp (need |%+.2f|>%.2f)",
                     movePip, movePip, InpReversalThresholdPip);
         lastVerbose = nowMs;
      }
   }
}

//+------------------------------------------------------------------+
