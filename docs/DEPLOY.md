# Deployment runbook (Google Cloud Run)

What runs where:

| Piece | Name | Notes |
|---|---|---|
| API | Cloud Run service `churn-api` | FastAPI + churn_core + saved models. Needs the API key. |
| Dashboard | Cloud Run service `churn-dashboard` | Streamlit. Calls the API with the key. |
| Images | Artifact Registry repo `churn` (asia-south1) | `api:vN` and `dashboard:vN` |
| API key | Secret Manager secret `churn-api-key` | Read by both services' own service accounts |
| Log DB login | Secret Manager secret `churn-log-url` | Neon connection string for the `api_writer` user. Read by `churn-api-sa` only |
| Database | Neon project (Singapore), database `neondb` | Customers, scores and the prediction log. Free plan: cannot bill, it pauses at its limits |
| Identities | `churn-api-sa`, `churn-dashboard-sa` | Each can read that one secret and nothing else |

Commands are for Windows `cmd`. Set these in each new terminal window:

```
set PROJECT=project-54e1d99f-6848-430a-886
set REGION=asia-south1
set REPO=%REGION%-docker.pkg.dev/%PROJECT%/churn
```

## One-time setup (already done)

```
gcloud artifacts repositories create churn --repository-format=docker --location=%REGION%
gcloud auth configure-docker %REGION%-docker.pkg.dev

gcloud iam service-accounts create churn-api-sa --display-name="Churn API runtime"
gcloud iam service-accounts create churn-dashboard-sa --display-name="Churn dashboard runtime"

.venv\Scripts\python -c "import secrets,sys; sys.stdout.write(secrets.token_urlsafe(32))" | gcloud secrets create churn-api-key --data-file=-
gcloud secrets add-iam-policy-binding churn-api-key --member="serviceAccount:churn-api-sa@%PROJECT%.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding churn-api-key --member="serviceAccount:churn-dashboard-sa@%PROJECT%.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

First deployments (v1):

```
docker build -f docker/Dockerfile.api -t %REPO%/api:v1 .
docker push %REPO%/api:v1
gcloud run deploy churn-api --image %REPO%/api:v1 --region %REGION% --service-account churn-api-sa@%PROJECT%.iam.gserviceaccount.com --set-secrets API_KEY=churn-api-key:latest --memory 1Gi --cpu 1 --cpu-boost --max-instances 1 --min-instances 0 --concurrency 10 --timeout 300 --port 8080 --allow-unauthenticated
gcloud run services update churn-api --region %REGION% --max 1

set URL=<the API's https address>
docker build -f docker/Dockerfile.dashboard -t %REPO%/dashboard:v1 .
docker push %REPO%/dashboard:v1
gcloud run deploy churn-dashboard --image %REPO%/dashboard:v1 --region %REGION% --service-account churn-dashboard-sa@%PROJECT%.iam.gserviceaccount.com --set-env-vars API_URL=%URL% --set-secrets API_KEY=churn-api-key:latest --memory 1Gi --cpu 1 --cpu-boost --max 1 --concurrency 10 --timeout 300 --port 8080 --allow-unauthenticated
```

## Database (Neon) and the prediction log

One-time setup (already done), in this order:
1. `sql/schema.sql` in the Neon SQL Editor (fresh database).
2. `python db/load_data.py --target neon` (asks for `yes`, runs ten checks, rolls back if any fails).
3. `sql/roles_neon.sql` with two generated passwords: `dashboard_ro` (read-only) and `api_writer`
   (the prediction log only). Prove the permissions with `python scripts/check_db_roles.py`.
4. The writer connection string goes to Secret Manager and only to the API:
   ```
   .venv\Scripts\python -c "from dotenv import load_dotenv; import os, sys; load_dotenv(); sys.stdout.write(os.environ['DATABASE_URL_WRITER'])" | gcloud secrets create churn-log-url --data-file=-
   gcloud secrets add-iam-policy-binding churn-log-url --member="serviceAccount:churn-api-sa@%PROJECT%.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
   gcloud run deploy churn-api --image %REPO%/api:v2 --region %REGION% --update-secrets PREDICTION_LOG_URL=churn-log-url:latest
   ```

How the log behaves: single predictions only, no names or IDs, at most 30 writes a minute, the newest
5,000 rows kept, every failure swallowed, a one-minute pause after a failure. It is off unless
`PREDICTION_LOG_URL` is set, so never put that variable in `.env` for everyday local work.
Check it is on: `/model-info` shows `"prediction_log":{"enabled":true}`. Neon sleeps after five idle
minutes, so the first prediction after a pause may not be logged. That is expected.

Reload data into Neon after a retrain: regenerate `reports/scores.csv`, then rerun the loader with
`--target neon`. It upserts, so it is safe to repeat.

## Ship a new version

1. Run `ruff check` and `pytest`, and check the change with `docker compose up --build`.
2. Build and push with a NEW tag (never reuse one), for example v2:
   ```
   docker build -f docker/Dockerfile.api -t %REPO%/api:v2 .
   docker push %REPO%/api:v2
   ```
3. Deploy it. The other settings (memory, limits, secrets, `--update-secrets` additions) carry over from the previous revision:
   ```
   gcloud run deploy churn-api --image %REPO%/api:v2 --region %REGION%
   ```
   Same idea for `churn-dashboard` with `dashboard:v2`.
4. Verify: `/health` answers, `/model-info` without a key returns 401, and the scale check below still shows 1.

**Roll back** to an earlier revision:
```
gcloud run revisions list --service churn-api --region %REGION%
gcloud run services update-traffic churn-api --to-revisions=REVISION_NAME=100 --region %REGION%
```

## Cost safety (do not remove these)

- `--max 1` at the service level on both services. Check it after every deploy:
  ```
  gcloud run services describe churn-api --region %REGION% --format=yaml | findstr /i scale
  gcloud run services describe churn-dashboard --region %REGION% --format=yaml | findstr /i scale
  ```
  Every `maxScale` line should say `'1'`.
- Minimum instances stay at 0. A minimum instance is billed while idle.
- `--timeout 300` cuts any request, including a held-open dashboard connection, after five minutes.
- Budget emails: `Churn-Budget` (counts spend after credits) and `Churn-Gross` (counts usage before credits). They only send emails; they do not stop spending.
- Check Billing > Reports, grouped by product, weekly. Free credits expire on 21 January 2027; after that the account is pay-as-you-go.
- Never print or paste the API key. To read it for a test without showing it:
  ```
  for /f %i in ('gcloud secrets versions access latest --secret=churn-api-key') do @set KEY=%i
  ... use %KEY% ...
  set KEY=
  ```

## Take the demo offline without deleting it

Make the dashboard private (visitors get an error, nothing is deleted):
```
gcloud run services remove-iam-policy-binding churn-dashboard --region %REGION% --member=allUsers --role=roles/run.invoker
```
Make it public again:
```
gcloud run services add-iam-policy-binding churn-dashboard --region %REGION% --member=allUsers --role=roles/run.invoker
```

## Full teardown

```
gcloud run services delete churn-dashboard --region %REGION%
gcloud run services delete churn-api --region %REGION%
gcloud artifacts repositories delete churn --location %REGION%
gcloud secrets delete churn-api-key
gcloud iam service-accounts delete churn-api-sa@%PROJECT%.iam.gserviceaccount.com
gcloud iam service-accounts delete churn-dashboard-sa@%PROJECT%.iam.gserviceaccount.com
```

Last resort, which shuts down everything in the project: `gcloud projects delete %PROJECT%`