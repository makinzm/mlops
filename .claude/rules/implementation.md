以下の順番で作業をしてください。

1. main branchで作業せずに、feature branchを切って作業してください。
2. まず、実行計画を docs/tasks/YYYY/Q/MMDD_<title>/README.md に書いてください。
3. 実行計画が承認されたら、テストが自動実行されるか lefthookおよび Github Actionsの確認をして、自動実行されるようにしてください。完了したら commitし報告してください。
4. 次にテストの実装のみを行って、エラーになることを確認し、そのログを docs/tasks/YYYY/Q/MMDD_<title>/TEST_LOG_YYYYMMDD_HHMMSS.md に保存してください。完了したら commitし報告してください。
5. 最後に実装を行い、テストが成功することを確認してください。完了したら commitし報告してください。
6. その後に、当該作業について人が手動で行う手順がある場合は、docs/manual/以下に手順をドキュメント化してください。完了したら commitし報告してください。
7. 最後に、ghコマンドを使用して、Pull Requestを作成してください。完了したら報告してください。
