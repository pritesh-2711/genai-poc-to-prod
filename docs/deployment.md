# AWS Deployment Guide (PILOT) — Step by Step

This guide takes you from zero to a pilot deployment on AWS. Read it end-to-end before starting — several steps depend on values you create in earlier steps.

---

## What you will build

```
Internet
  │
  ▼
CloudFront (CDN)          ← serves the React frontend from S3
  │
  ▼
S3 (frontend bucket)      ← static Vite build output

Internet
  │
  ▼
Application Load Balancer ← HTTPS termination, routes to backend
  │
  ▼
ECS Fargate (backend)     ← FastAPI container from ECR
  │
  ├── RDS PostgreSQL       ← pgvector database
  ├── ElastiCache Redis    ← LangGraph checkpointer + clarification state
  └── S3 (documents)      ← uploaded PDF/DOCX files
```

**Estimated AWS monthly cost** for a small deployment (1 Fargate task, db.t3.micro, cache.t3.micro):
~$60–100/month.
RDS and ElastiCache are the largest cost items.

---

## Prerequisites

- An AWS account (free tier is not sufficient — RDS and ECS Fargate cost money)
- AWS CLI installed locally: `aws --version`
- Docker installed locally (you already have this)
- The repo pushed to GitHub

---

## Part 1 — One-time AWS infrastructure setup

Do these steps once. After they are done, every `git push` to `main` deploys automatically.

---

### Step 1 — Create an IAM user for GitHub Actions

GitHub Actions needs AWS credentials to push Docker images and deploy.

1. Open AWS Console → **IAM** → **Users** → **Create user**
2. Username: `github-actions-poc2prod`
3. Select **"Attach policies directly"** and attach these managed policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
   - `AmazonS3FullAccess`
   - `CloudFrontFullAccess`
4. Click **Create user**
5. Open the user → **Security credentials** → **Create access key**
6. Choose **"Application running outside AWS"**
7. **Save the Access Key ID and Secret Access Key** — you only see the secret once.

> You will add these to GitHub Secrets in Part 2.

---

### Step 2 — Create an ECR repository (backend image)

ECR is AWS's private Docker registry — your backend images are stored here.

```bash
aws ecr create-repository \
  --repository-name poc2prod-backend \
  --region us-east-1          # change to your preferred region
```

Save the `repositoryUri` from the output — it looks like:
`123456789.dkr.ecr.us-east-1.amazonaws.com/poc2prod-backend`

---

### Step 3 — Create a VPC (or use the default)

ECS, RDS, and ElastiCache all need to be in a VPC. The **default VPC** in your account is fine for a first deployment.

Find your default VPC:
```bash
aws ec2 describe-vpcs --filters "Name=is-default,Values=true" \
  --query "Vpcs[0].{VpcId:VpcId}" --output text
```

Find its subnet IDs (save them — you need at least 2 subnets in different AZs):
```bash
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=<your-vpc-id>" \
  --query "Subnets[*].{ID:SubnetId,AZ:AvailabilityZone}" \
  --output table
```

---

### Step 4 — Create RDS PostgreSQL with pgvector

1. Open AWS Console → **RDS** → **Create database**
2. Settings:
   - Engine: **PostgreSQL**
   - Engine version: **16.x** (pgvector requires pg12+)
   - Template: **Free tier** (or Production for production use)
   - DB instance identifier: `poc2prod-db`
   - Master username: `postgres`
   - Master password: choose a strong password and save it
   - Instance class: `db.t3.micro` (cheapest)
   - Storage: 20 GB, gp2
   - **VPC**: select your default VPC
   - **Public access**: **No** (ECS talks to it privately; you can use SSM/bastion for admin)
   - Create a new security group named `poc2prod-db-sg`
3. Click **Create database** — takes ~5 minutes

**Enable pgvector** — after the database is running, connect via the bastion method below and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Run the schema** — connect to the DB and run the contents of `pilot/sql/init.sql`:
```bash
# Option: temporarily enable Public access in RDS settings, then:
psql -h <rds-endpoint> -U postgres -d postgres -f pilot/sql/init.sql
# Then disable Public access again
```

Save the RDS endpoint — it looks like: `poc2prod-db.xxxxx.us-east-1.rds.amazonaws.com`

---

### Step 5 — Create ElastiCache Redis

Redis is used by LangGraph for conversation state and the clarification store when `storage.deployment: cloud`.

1. Open AWS Console → **ElastiCache** → **Create cluster**
2. Settings:
   - Cluster mode: **Disabled** (simpler, sufficient for this app)
   - Name: `poc2prod-redis`
   - Engine: Redis OSS
   - Node type: `cache.t3.micro`
   - Replicas: 0 (for cost; add 1 for production reliability)
   - **VPC**: your default VPC
   - Subnet group: create new, select all subnets
   - Security group: create new `poc2prod-redis-sg`
3. Click **Create**

Save the **Primary endpoint** — looks like:
`poc2prod-redis.xxxxx.cache.amazonaws.com:6379`

Your Redis URL will be: `redis://poc2prod-redis.xxxxx.cache.amazonaws.com:6379`

---

### Step 6 — Create S3 bucket for document uploads

```bash
aws s3 mb s3://poc2prod-documents-<your-unique-suffix> --region us-east-1
```

Save the bucket name.

---

### Step 7 — Create security groups and open ports

ECS tasks need to reach RDS and Redis. Open the correct ports:

**Allow ECS → RDS (PostgreSQL port 5432):**
1. Open **EC2 → Security Groups** → find `poc2prod-db-sg`
2. **Inbound rules** → Add rule:
   - Type: PostgreSQL, Port: 5432
   - Source: the security group that your ECS tasks will use (you'll know this after Step 9; come back and update it)

**Allow ECS → Redis (port 6379):**
1. Find `poc2prod-redis-sg`
2. Inbound rules → Add rule:
   - Type: Custom TCP, Port: 6379
   - Source: ECS security group

---

### Step 8 — Store secrets in AWS Secrets Manager

Never put database passwords or API keys in the ECS task definition in plaintext. Store them in Secrets Manager and inject them as environment variables.

```bash
# Store all backend secrets as a single JSON secret
aws secretsmanager create-secret \
  --name poc2prod/backend \
  --secret-string '{
    "OPENAI_API_KEY": "sk-...",
    "JWT_SECRET_KEY": "<run: python -c \"import secrets; print(secrets.token_hex(32))\">",
    "DB_HOST": "poc2prod-db.xxxxx.us-east-1.rds.amazonaws.com",
    "DB_PORT": "5432",
    "DB_NAME": "poc2prod",
    "DB_USER": "postgres",
    "DB_PASSWORD": "your-rds-password",
    "AWS_S3_BUCKET": "poc2prod-documents-<suffix>",
    "AWS_S3_REGION": "us-east-1",
    "REDIS_URL": "redis://poc2prod-redis.xxxxx.cache.amazonaws.com:6379",
    "STORAGE_DEPLOYMENT": "cloud"
  }'
```

Save the secret ARN from the output — looks like:
`arn:aws:secretsmanager:us-east-1:123456789:secret:poc2prod/backend-xxxxxx`

---

### Step 9 — Create an ECS cluster

```bash
aws ecs create-cluster --cluster-name poc2prod
```

---

### Step 10 — Create an IAM role for ECS tasks

ECS tasks need permission to read from Secrets Manager and write to S3.

1. Open **IAM** → **Roles** → **Create role**
2. Trusted entity type: **AWS service** → Use case: **Elastic Container Service Task**
3. Attach these policies:
   - `SecretsManagerReadWrite` (or create a custom policy scoped to your secret)
   - `AmazonS3FullAccess`
4. Role name: `poc2prod-ecs-task-role`

Also create an **ECS task execution role** (for pulling the image and writing logs):
1. Same process, Trusted entity: ECS Task
2. Attach: `AmazonECSTaskExecutionRolePolicy` + `SecretsManagerReadWrite`
3. Role name: `poc2prod-ecs-execution-role`

---

### Step 11 — Create the ECS task definition

This defines the backend container — how much CPU/memory, which image, which secrets.

Create a file `pilot/docs/task-definition.json` (**do not commit this file** — it contains ARNs specific to your AWS account):

```json
{
  "family": "poc2prod-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/poc2prod-ecs-execution-role",
  "taskRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/poc2prod-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/poc2prod-backend:latest",
      "portMappings": [
        { "containerPort": 8000, "protocol": "tcp" }
      ],
      "secrets": [
        { "name": "OPENAI_API_KEY",   "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:OPENAI_API_KEY::" },
        { "name": "JWT_SECRET_KEY",   "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:JWT_SECRET_KEY::" },
        { "name": "DB_HOST",          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:DB_HOST::" },
        { "name": "DB_PORT",          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:DB_PORT::" },
        { "name": "DB_NAME",          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:DB_NAME::" },
        { "name": "DB_USER",          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:DB_USER::" },
        { "name": "DB_PASSWORD",      "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:DB_PASSWORD::" },
        { "name": "AWS_S3_BUCKET",    "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:AWS_S3_BUCKET::" },
        { "name": "AWS_S3_REGION",    "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:AWS_S3_REGION::" },
        { "name": "REDIS_URL",        "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:REDIS_URL::" },
        { "name": "STORAGE_DEPLOYMENT", "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:poc2prod/backend-xxxxxx:STORAGE_DEPLOYMENT::" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/poc2prod-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

Register it:
```bash
aws ecs register-task-definition \
  --cli-input-json file://pilot/docs/task-definition.json
```

Create a CloudWatch log group for the logs:
```bash
aws logs create-log-group --log-group-name /ecs/poc2prod-backend
```

---

### Step 12 — Create an Application Load Balancer (ALB)

The ALB receives HTTPS traffic from the internet and forwards it to ECS.

1. Open **EC2** → **Load Balancers** → **Create Load Balancer** → **Application Load Balancer**
2. Settings:
   - Name: `poc2prod-alb`
   - Scheme: **Internet-facing**
   - IP address type: IPv4
   - VPC: default VPC, select all available subnets
   - Security group: create new `poc2prod-alb-sg` with inbound HTTP (80) and HTTPS (443) open to `0.0.0.0/0`
3. Listener: **HTTPS :443** → you need an ACM certificate (see below)
4. Target group: Create new
   - Type: IP addresses
   - Name: `poc2prod-backend-tg`
   - Protocol: HTTP, Port: 8000
   - Health check path: `/health`

**Get an SSL certificate with ACM:**
- If you have a domain (e.g. `api.yourdomain.com`): go to **ACM** → **Request certificate** → enter your domain → validate via DNS (add CNAME to your DNS provider)
- If you don't have a domain yet: add an HTTP-only listener on port 80 for now and switch to HTTPS later

After creating the ALB, save the **DNS name** — looks like:
`poc2prod-alb-123456.us-east-1.elb.amazonaws.com`

---

### Step 13 — Create the ECS service

```bash
aws ecs create-service \
  --cluster poc2prod \
  --service-name poc2prod-backend \
  --task-definition poc2prod-backend \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[subnet-aaa,subnet-bbb],
    securityGroups=[sg-your-ecs-sg],
    assignPublicIp=ENABLED
  }" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=backend,containerPort=8000" \
  --health-check-grace-period-seconds 120
```

After the service is running, go back to **Step 7** and add the ECS security group as a source in the RDS and Redis inbound rules.

---

### Step 14 — Create the frontend S3 bucket and CloudFront distribution

**S3 bucket for frontend:**
```bash
aws s3 mb s3://poc2prod-frontend-<unique-suffix> --region us-east-1

# Block all public access (CloudFront will serve it, not direct S3 URLs)
aws s3api put-public-access-block \
  --bucket poc2prod-frontend-<unique-suffix> \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**CloudFront distribution:**
1. Open **CloudFront** → **Create distribution**
2. Origin domain: select your `poc2prod-frontend-*` S3 bucket
3. Origin access: **Origin access control (OAC)** → Create new OAC → update S3 bucket policy (CloudFront will show you the policy JSON to paste)
4. Default root object: `index.html`
5. Error pages: Add custom error response → HTTP 403 → Response path `/index.html` → Response code 200 (this makes SPA routing work — `try_files` equivalent for CloudFront)
6. If you have a domain: add it in **Alternate domain names** + attach ACM cert
7. Click **Create distribution** — takes ~10 minutes to deploy globally

Save the **CloudFront distribution ID** and the **CloudFront domain** (e.g. `d1234abcd.cloudfront.net`).

---

### Step 15 — Update CORS in the backend

The FastAPI app currently only allows `localhost` origins. Add your production frontend URL.

Edit [pilot/src/api/main.py](../src/api/main.py):

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        "https://d1234abcd.cloudfront.net",    # your CloudFront domain
        "https://yourdomain.com",              # if you have a custom domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Step 16 — Update config.yaml to support cloud storage via env var

The `configs/config.yaml` has `storage.deployment: local` hardcoded. In production, ECS injects `STORAGE_DEPLOYMENT=cloud` via Secrets Manager. You need to make this field read from an env var.

Edit [pilot/configs/config.yaml](../configs/config.yaml) — change the storage block:

```yaml
storage:
  deployment: "${STORAGE_DEPLOYMENT:-local}"   # env var override; falls back to local
  cloud:
    provider: aws
    aws:
      s3_bucket: "${AWS_S3_BUCKET}"
      s3_region: "${AWS_S3_REGION}"
      redis_url: "${REDIS_URL}"
```

With this change:
- Local dev: no `STORAGE_DEPLOYMENT` env var → uses `local` (disk + in-process state)
- Production ECS: `STORAGE_DEPLOYMENT=cloud` injected from Secrets Manager → uses S3 + Redis

---

## Part 2 — GitHub repository setup

### Step 17 — Add GitHub Secrets

Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

Add each of these:

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Access key from Step 1 |
| `AWS_SECRET_ACCESS_KEY` | Secret key from Step 1 |
| `AWS_REGION` | e.g. `us-east-1` |
| `BACKEND_ECR_REPOSITORY` | `poc2prod-backend` |
| `ECS_CLUSTER` | `poc2prod` |
| `BACKEND_ECS_SERVICE` | `poc2prod-backend` |
| `BACKEND_TASK_DEFINITION_FAMILY` | `poc2prod-backend` |
| `BACKEND_CONTAINER_NAME` | `backend` |
| `FRONTEND_S3_BUCKET` | `poc2prod-frontend-<suffix>` |
| `CLOUDFRONT_DISTRIBUTION_ID` | From Step 14 |
| `VITE_API_BASE_URL` | `https://api.yourdomain.com` or `https://poc2prod-alb-123456.us-east-1.elb.amazonaws.com` |

---

## Part 3 — First deployment

### Step 18 — Push to main and trigger CI/CD

After completing Steps 15 and 16 (CORS + config.yaml changes):

```bash
git add .
git commit -m "add production deployment config"
git push origin main
```

Go to your GitHub repo → **Actions** tab. You will see two workflows running:
- `Backend — Build & Deploy`
- `Frontend — Build & Deploy`

The backend workflow takes ~8–12 minutes (building the Docker image with LibreOffice etc. takes a while). The frontend workflow takes ~2 minutes.

### Step 19 — Verify the deployment

**Backend health:**
```bash
curl https://your-alb-domain/health
# Expected: {"status": "ok"}
```

**Frontend:**
Open `https://d1234abcd.cloudfront.net` in your browser. You should see the login screen.

**Sign up and approve your account:**
```bash
# Connect to RDS through a bastion host or temporarily enable public access
psql -h <rds-endpoint> -U postgres -d poc2prod

UPDATE poc2prod.users SET status = 'approved' WHERE email = 'you@example.com';
```

---

## Updating secrets later

If you need to rotate a secret (e.g. new OpenAI API key):

```bash
aws secretsmanager update-secret \
  --secret-id poc2prod/backend \
  --secret-string '{ "OPENAI_API_KEY": "sk-new-key", ... all other keys unchanged ... }'
```

Then force a new ECS deployment to pick up the updated secret:
```bash
aws ecs update-service \
  --cluster poc2prod \
  --service poc2prod-backend \
  --force-new-deployment
```

---

## Manual workflow trigger

You can redeploy without pushing code from the GitHub UI:
1. Go to **Actions** → select the workflow
2. Click **Run workflow** → **Run workflow**

Or via CLI:
```bash
gh workflow run backend.yml
gh workflow run frontend.yml
```

---

## Rollback

Each Docker image is tagged with the git commit SHA. To roll back to a previous version:

```bash
# Find the previous commit SHA
git log --oneline -5

# Update the ECS service to use the old image
aws ecs update-service \
  --cluster poc2prod \
  --service poc2prod-backend \
  --task-definition poc2prod-backend:<previous-revision-number>
```

Or re-run a previous GitHub Actions workflow run — each run pushed an image tagged with its commit SHA.

---

## Useful commands

```bash
# Check ECS service status
aws ecs describe-services --cluster poc2prod --services poc2prod-backend \
  --query "services[0].{Status:status,Running:runningCount,Desired:desiredCount}"

# Watch ECS deployment events
aws ecs describe-services --cluster poc2prod --services poc2prod-backend \
  --query "services[0].events[:5]"

# View backend logs (last 50 lines)
aws logs tail /ecs/poc2prod-backend --follow

# List all images in ECR (to find SHA tags for rollback)
aws ecr list-images --repository-name poc2prod-backend \
  --query "imageIds[*].imageTag" --output table

# Force redeploy without code change
aws ecs update-service --cluster poc2prod --service poc2prod-backend \
  --force-new-deployment
```
