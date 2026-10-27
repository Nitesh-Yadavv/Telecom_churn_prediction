# Deployment runbook (Google Cloud Run)

What runs where:

| Piece | Name | Notes |
|---|---|---|
| API | Cloud Run service `churn-api` | FastAPI + churn_core + saved models. Needs the API key. |
| Dashboard | Cloud Run service `churn-dashboard` | Streamlit. Calls the API with the key. |
| Images | Artifact Registry repo `churn` (asia-south1) | `api:vN` and `dashboard:vN` |
| API key | Secret Manager secret `churn-api-key` | Read by both services' own service accounts |
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

## Ship a new version

1. Run `ruff check` and `pytest`, and check the change with `docker compose up --build`.
2. Build and push with a NEW tag (never reuse one), for example v2:
   ```
   docker build -f docker/Dockerfile.api -t %REPO%/api:v2 .
   docker push %REPO%/api:v2
   ```
3. Deploy it. The other settings (memory, limits, secrets) carry over from the previous revision:
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