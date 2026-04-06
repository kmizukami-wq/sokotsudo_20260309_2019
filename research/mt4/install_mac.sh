#!/bin/bash
# BB_Reversal_Martin EA - Mac MT4 自動インストールスクリプト
# 使い方: ターミナルにこの1行を貼り付けてEnter
# curl -sL https://raw.githubusercontent.com/kmizukami-wq/sokotsudo/claude/usdjpy-trading-strategy-mVUws/research/mt4/install_mac.sh | bash

echo ""
echo "================================================"
echo "  BB_Reversal_Martin EA インストーラー (Mac)"
echo "================================================"
echo ""

# MT4のデータフォルダを探す
MT4_DIRS=$(find ~/Library/Application\ Support -maxdepth 2 -name "MQL4" -type d 2>/dev/null)

if [ -z "$MT4_DIRS" ]; then
    # 別の場所も探す
    MT4_DIRS=$(find ~/Library -maxdepth 4 -name "MQL4" -type d 2>/dev/null)
fi

if [ -z "$MT4_DIRS" ]; then
    echo "MT4のフォルダが見つかりません。"
    echo "MT4を起動して「ファイル」→「データフォルダを開く」でパスを確認してください。"
    echo ""
    read -p "MT4のデータフォルダのパスを入力: " MT4_PATH
    EXPERTS_DIR="$MT4_PATH/MQL4/Experts"
else
    # 最初に見つかったものを使用
    EXPERTS_DIR="$(echo "$MT4_DIRS" | head -1)/Experts"
    echo "MT4フォルダ検出: $EXPERTS_DIR"
fi

# Expertsフォルダがなければ作成
mkdir -p "$EXPERTS_DIR" 2>/dev/null

if [ ! -d "$EXPERTS_DIR" ]; then
    echo "エラー: $EXPERTS_DIR を作成できません"
    exit 1
fi

# EAファイルをダウンロード
EA_URL="https://raw.githubusercontent.com/kmizukami-wq/sokotsudo/claude/usdjpy-trading-strategy-mVUws/research/mt4/BB_Reversal_Martin.mq4"
EA_FILE="$EXPERTS_DIR/BB_Reversal_Martin.mq4"

echo "ダウンロード中..."
curl -sL "$EA_URL" -o "$EA_FILE"

if [ -f "$EA_FILE" ]; then
    echo ""
    echo "インストール完了!"
    echo "ファイル: $EA_FILE"
    echo ""
    echo "================================================"
    echo "  次のステップ"
    echo "================================================"
    echo ""
    echo "1. MT4を再起動（または ナビゲータ → Expert Advisors → 右クリック → 更新）"
    echo "2. 「BB_Reversal_Martin」が表示されることを確認"
    echo "3. 通貨ペアのM15チャートを開く"
    echo "4. EAをチャートにドラッグ＆ドロップ"
    echo "5. MagicNumberをペア別に設定:"
    echo ""
    echo "   AUD/JPY → 10001"
    echo "   AUD/USD → 10002"
    echo "   EUR/USD → 10003"
    echo "   GBP/USD → 10004"
    echo "   EUR/JPY → 10005"
    echo "   USD/JPY → 10006"
    echo "   NZD/USD → 10007"
    echo ""
    echo "※ まずデモ口座で1〜2週間テストしてください!"
    echo ""
else
    echo "エラー: ダウンロードに失敗しました"
    echo "手動でダウンロード: $EA_URL"
    exit 1
fi
