// FieldOps on Azure Container Apps.
//
//   az group create -n fieldops-staging -l uksouth
//   az deployment group create -g fieldops-staging -f infra/azure/main.bicep \
//     -p env=staging acrName=<registry> image=<registry>.azurecr.io/fieldops:<sha> \
//        databaseUrl=<secret> redisHost=<host:6380> redisPassword=<secret> \
//        openaiApiKey=<secret> langsmithApiKey=<secret> apiKeys=<key:tenant,...>
//
// Creates: Log Analytics + Application Insights (OpenTelemetry target), a Container Apps
// environment, a user-assigned identity with AcrPull, the API app (external ingress, probes,
// HTTP autoscaling) and the Celery worker (no ingress, KEDA autoscaling on Redis queue depth).
// Postgres (Flexible Server + pgvector) and Redis (Azure Cache) are passed in as connection
// details; in a real setup they are separate modules with private networking.

@allowed(['staging', 'prod'])
param env string
param location string = resourceGroup().location
param acrName string
param image string
@secure()
param databaseUrl string
param redisHost string
@secure()
param redisPassword string
@secure()
param openaiApiKey string
@secure()
param langsmithApiKey string = ''
@description('API key to tenant map, "key1:t1,key2:t2". Must include the smoke-test tenant key.')
@secure()
param apiKeys string
param promptTag string = env == 'prod' ? 'prod' : 'staging'

var prefix = 'fieldops-${env}'
var redisUrl = 'rediss://:${redisPassword}@${redisHost}/0'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-ai'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-id'
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

// AcrPull, so the apps pull images with the managed identity instead of registry passwords.
resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, 'acrpull')
  scope: acr
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  }
}

resource cae 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var secrets = [
  { name: 'database-url', value: databaseUrl }
  { name: 'redis-url', value: redisUrl }
  { name: 'redis-password', value: redisPassword }
  { name: 'openai-api-key', value: openaiApiKey }
  { name: 'langsmith-api-key', value: empty(langsmithApiKey) ? 'unset' : langsmithApiKey }
  { name: 'api-keys', value: apiKeys }
  { name: 'appinsights', value: appInsights.properties.ConnectionString }
]

var commonEnv = [
  { name: 'DATABASE_URL', secretRef: 'database-url' }
  { name: 'REDIS_URL', secretRef: 'redis-url' }
  { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
  { name: 'LANGSMITH_API_KEY', secretRef: 'langsmith-api-key' }
  { name: 'LANGSMITH_TRACING', value: empty(langsmithApiKey) ? 'false' : 'true' }
  { name: 'LANGSMITH_PROJECT', value: 'fieldops-${env}' }
  { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights' }
  { name: 'FIELDOPS_PROMPT_TAG', value: promptTag }
  { name: 'FIELDOPS_PROMPT_SOURCE', value: 'langsmith' }
  { name: 'FIELDOPS_API_KEYS', secretRef: 'api-keys' }
]

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [ acrPull ]
  properties: {
    managedEnvironmentId: cae.id
    configuration: {
      activeRevisionsMode: 'Multiple'  // lets the deploy workflow smoke-test before shifting traffic
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [ { server: acr.properties.loginServer, identity: identity.id } ]
      secrets: secrets
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: commonEnv
          probes: [
            { type: 'Liveness', httpGet: { path: '/healthz', port: 8000 }, periodSeconds: 15 }
            { type: 'Readiness', httpGet: { path: '/readyz', port: 8000 }, periodSeconds: 10, failureThreshold: 3 }
            { type: 'Startup', httpGet: { path: '/healthz', port: 8000 }, failureThreshold: 30, periodSeconds: 2 }
          ]
        }
      ]
      scale: {
        minReplicas: env == 'prod' ? 2 : 1   // no scale-to-zero: voice can't absorb cold starts
        maxReplicas: env == 'prod' ? 10 : 2
        rules: [ { name: 'http', http: { metadata: { concurrentRequests: '30' } } } ]
      }
    }
  }
}

resource worker 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-worker'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [ acrPull ]
  properties: {
    managedEnvironmentId: cae.id
    configuration: {
      activeRevisionsMode: 'Single'
      registries: [ { server: acr.properties.loginServer, identity: identity.id } ]
      secrets: secrets
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: image
          command: [ 'celery', '-A', 'capstone.workers.celery_app', 'worker', '--loglevel=INFO', '--concurrency=4' ]
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: commonEnv
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: env == 'prod' ? 20 : 3
        rules: [
          {
            name: 'queue-depth'
            custom: {
              type: 'redis'
              metadata: {
                address: redisHost
                listName: 'inbound'
                listLength: '20'
                enableTLS: 'true'
              }
              auth: [ { secretRef: 'redis-password', triggerParameter: 'password' } ]
            }
          }
        ]
      }
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output appInsightsName string = appInsights.name
