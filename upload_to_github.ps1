# Run in PowerShell from the folder that contains gkx-replication (e.g. C:\Users\dell\Desktop\5054)
# Replace the e-mail below with the one of your GitHub account.
cd .\gkx-replication
git init
git config user.name "Zhang Qi"
git config user.email "your-github-email@example.com"
git add .
git commit -m "GKX replication: code, notebooks and final results"
git branch -M main
git remote add origin https://github.com/ZQ-77777/gkx-replication.git
git push -u origin main
# If the push is rejected because the GitHub repository already has an older version, run:
#   git push -u origin main --force
