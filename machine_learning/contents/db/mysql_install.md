# MySQLの環境構築

- MySQL Community Serverのインストール
- MySQL Shellのインストール
- Visual Studio Codeのインストール

## Mac

### Macに利用されているCPUアーキテクチャを確認する

1. Spotlightを開く（Command + Spaceを押す）
2. "ターミナル"と入力してEnterを押す
3. ターミナルで次のコマンドを実行する:

```bash
uname -m
```

- `arm64`が表示された場合、ARMアーキテクチャを利用しています。
- `x86_64`が表示された場合、Intelアーキテクチャを利用しています。

### MySQL Community Serverのインストール

1. [こちら](https://dev.mysql.com/downloads/)のページからMySQL Community Serverにアクセスします。
2. CPUアーキテクチャに合わせてDMGファイルをダウンロードします。
3. "No thanks, just start my download."をクリックします。
4. ダウンロードしたDMGファイルを実行し、インストールを開始します。
5. "Please enter the password for root user"と表示されたら、rootユーザのパスワードを入力します。**パスワードは忘れないようにしてください。**

### MySQL Shellのインストール

1. [こちら](https://dev.mysql.com/downloads/)のページからMySQL Shellにアクセスします。
2. CPUアーキテクチャに合わせてDMGファイルをダウンロードします。
3. `No thanks, just start my download.`をクリックします。
4. ダウンロードファイルを実行し、インストールを開始します。

### Visual Studio Codeのインストール


### Extensionのインストール

1. `MySQL Shell`を検索し、`MySQL Shell for VS Code`をインストールします。
2. `Next`をクリックします。
3. `Reload VS Code Window`をクリックします。



## Windows

## DB Connection

1. 