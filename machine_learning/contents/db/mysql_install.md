# MySQLの環境構築

## Mac

### Macに利用されているCPUアーキテクチャを確認する

1. Spotlightを開く（Command + Space）
2. "ターミナル"と入力してEnter
3. `uname -m`のコマンドを実行する

`arm64` -> ARMアーキテクチャ
`x86_64` -> Intelアーキテクチャ

### MySQL Community Serverのインストール

1. [こちら](https://dev.mysql.com/downloads/)のページからMySQL Community Serverにアクセスします。
2. CPUアーキテクチャに合わせてDMGファイルをダウンロードします。
3. `No thanks, just start my download.`をクリックします。
4. ダウンロードファイルを実行し、インストールを開始します。
5. `please enter the password for root user`と表示されたら、rootユーザのパスワードを入力します。パスワードは忘れないようにしてください。

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