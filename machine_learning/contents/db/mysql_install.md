# MySQLの環境構築

- MySQL Community Serverのインストール
- MySQL Shellのインストール
- Visual Studio Codeのインストール

## Mac

### Macに利用されているCPUアーキテクチャを確認する

1. Spotlightを開く（`Command + Space`を押す）
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
5. "Please enter the password for root user"と表示されたら、rootユーザの**パスワード**を入力します。

Step 5で入力したパスワードは、MySQLのrootユーザのパスワードです。このパスワードは忘れないようにしてください。

### MySQL Shellのインストール

1. [こちら](https://dev.mysql.com/downloads/)のページからMySQL Shellにアクセスします。
2. CPUアーキテクチャに合わせてDMGファイルをダウンロードします。
3. "No thanks, just start my download."をクリックします。
4. ダウンロードしたDMGファイルを実行し、インストールを開始します。

## Windows

### MySQL Community Serverのインストール

### MySQL Shellのインストール

## Visual Studio Codeのインストール

1. [こちら](https://code.visualstudio.com/)のページからVisual Studio Codeにアクセスします。
2. "Download for Mac"または"Download for Windows"をクリックします。
3. ダウンロードしたファイルを実行し、インストールを開始します。

## MySQL Shell for VS Codeのインストール

1. Visual Studio Codeを開きます。
2. Extensionsを開きます。
   - Mac: `Command + Shift + X`
   - Windows: `Ctrl + Shift + X`
3. "MySQL Shell for VS Code"を検索し、インストールします。
4. インストールが完了したら、"Next"をクリックし、インストールの認証を行います。
5. "Reload VS Code Window"をクリックし、Visual Studio Codeを再起動します。

## DB Connection

1. Visual Studio Codeを開きます。
2. 左側のメニューから"MySQL Shell for VS Code"のアイコン🐬をクリックします。
3. 左側のメニューから"DATABASE CONNECTIONS"を見つけて、➕をクリックします。
4. `Caption`にconnectionの名前、`User Name`に`root`を入力します。
5. OKをクリックします。
6. 作成したconnectionをクリックし、接続します。
7. パスワードを入力します。
8. 接続が完了したら、`SHOW DATABASES;`を実行して、下記のデータベースが表示されたら接続成功です。
    
```bash
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
```
