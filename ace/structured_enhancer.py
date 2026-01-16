#!/usr/bin/env python
"""
Structured Query Enhancer based on .enhancedprompt.md methodology.

EXHAUSTIVE IMPLEMENTATION - Covers ALL software engineering domains.

Transforms vague user queries into structured, actionable prompts using:
1. Intent Classification (ANALYTICAL/IMPLEMENTATION/TROUBLESHOOTING/EXPLORATORY/LEARNING/REFACTORING)
2. Domain Detection (40+ domains covering all software engineering areas)
3. Context Expansion with domain-specific terminology (1000+ expansion terms)
4. Query Restructuring for optimal retrieval

This implements the EnginizeAPI prompt enhancement methodology.
"""

import os
import sys
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QueryIntent(Enum):
    """Query intent classification - EXHAUSTIVE."""
    ANALYTICAL = "analytical"           # compare, analyze, understand, investigate, evaluate
    IMPLEMENTATION = "implementation"   # create, modify, add, develop, code
    EXPLORATORY = "exploratory"         # what, how, why - seeking information
    TROUBLESHOOTING = "troubleshooting" # fix, broken, error, failing, debug
    LEARNING = "learning"               # learn, tutorial, example, getting started
    REFACTORING = "refactoring"         # improve, clean, optimize, modernize
    PLANNING = "planning"               # design, plan, architect, strategy
    REVIEWING = "reviewing"             # review, audit, check, validate


class QueryDomain(Enum):
    """Technical domain classification - EXHAUSTIVE (40+ domains)."""
    # === CORE SOFTWARE ENGINEERING ===
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DATABASE = "database"
    API = "api"
    CONFIGURATION = "configuration"
    
    # === DEVOPS & INFRASTRUCTURE ===
    DEVOPS = "devops"
    CI_CD = "ci_cd"
    CONTAINERS = "containers"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    LOGGING = "logging"
    
    # === FRONTEND ===
    FRONTEND = "frontend"
    CSS = "css"
    JAVASCRIPT = "javascript"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    ACCESSIBILITY = "accessibility"
    
    # === BACKEND ===
    BACKEND = "backend"
    PYTHON = "python"
    JAVA = "java"
    NODEJS = "nodejs"
    GOLANG = "golang"
    RUST = "rust"
    DOTNET = "dotnet"
    
    # === DATA & AI ===
    DATA_ENGINEERING = "data_engineering"
    MACHINE_LEARNING = "machine_learning"
    AI = "ai"
    ANALYTICS = "analytics"
    ETL = "etl"
    DATA_SCIENCE = "data_science"
    
    # === MESSAGING & EVENTS ===
    MESSAGING = "messaging"
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
    EVENTS = "events"
    STREAMING = "streaming"
    
    # === MOBILE ===
    MOBILE = "mobile"
    IOS = "ios"
    ANDROID = "android"
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"
    
    # === VERSION CONTROL ===
    GIT = "git"
    VERSION_CONTROL = "version_control"
    
    # === DOCUMENTATION & COMMUNICATION ===
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"
    
    # === SPECIFIC TECHNOLOGIES ===
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    
    # === QUALITY & COMPLIANCE ===
    CODE_QUALITY = "code_quality"
    COMPLIANCE = "compliance"
    GOVERNANCE = "governance"
    
    # === CATCH-ALL ===
    GENERAL = "general"


@dataclass
class EnhancedQuery:
    """Result of query enhancement."""
    original_query: str
    intent: QueryIntent
    domains: List[QueryDomain]
    enhanced_query: str
    expansion_terms: List[str]
    structured_prompt: str


# Intent detection patterns - EXHAUSTIVE
INTENT_PATTERNS = {
    QueryIntent.ANALYTICAL: [
        r'\b(compare|compar|versus|vs\.?)\b',
        r'\b(analyze|analyz|analysis|assess|assessment)\b',
        r'\b(understand|comprehend|grasp|fathom)\b',
        r'\b(investigate|investig|research|study|examine)\b',
        r'\b(evaluate|evaluat|appraise|gauge)\b',
        r'\b(review|audit|inspect|scrutinize)\b',
        r'\b(what.*differ|how.*relate|why.*design)\b',
        r'\b(pros.*cons|trade.?off|advantage|disadvantage)\b',
        r'\b(strengths?|weaknesses?|swot)\b',
        r'\b(correlation|relationship|impact|effect)\b',
        r'\b(root cause|cause.*effect|why.*happen)\b',
    ],
    QueryIntent.IMPLEMENTATION: [
        r'\b(create|implement|build|construct|develop)\b',
        r'\b(add|insert|include|incorporate|integrate)\b',
        r'\b(modify|change|update|alter|adjust)\b',
        r'\b(write|code|program|script|develop)\b',
        r'\b(make|produce|generate|craft)\b',
        r'\b(setup|set up|install|configure|deploy)\b',
        r'\b(connect|link|wire|hook)\b',
        r'\b(enable|activate|turn on|switch on)\b',
        r'\b(migrate|convert|transform|port)\b',
        r'\b(extend|expand|augment|enhance)\b',
    ],
    QueryIntent.TROUBLESHOOTING: [
        r'\b(fix|repair|resolve|solve|correct)\b',
        r'\b(broken|break|broke|fails?|failing)\b',
        r'\b(error|errors?|exception|exceptions?)\b',
        r'\b(issue|issues?|problem|problems?|bug|bugs?)\b',
        r'\b(crash|crashes?|crashe?d|crashed)\b',
        r'\b(not working|doesn.t work|does not work|won.t work)\b',
        r'\b(wrong|incorrect|invalid|unexpected)\b',
        r'\b(debug|troubleshoot|diagnose|investigate)\b',
        r'\b(hang|hangs?|hung|freeze|frozen|stuck)\b',
        r'\b(timeout|timed? out|slow|unresponsive)\b',
        r'\b(memory leak|leak|out of memory|oom)\b',
        r'\b(null|undefined|none|nil|missing)\b',
        r'\b(cannot|can.t|unable|impossible)\b',
        r'\b(401|403|404|500|502|503|504)\b',  # HTTP error codes
    ],
    QueryIntent.EXPLORATORY: [
        r'\b(what|which|where|when|who|whose)\b',
        r'\b(how|how to|howto)\b',
        r'\b(why|reason|purpose)\b',
        r'\b(explain|describe|show|tell|clarify)\b',
        r'\b(mean|means|meaning|definition|define)\b',
        r'\b(difference|differ|distinction|distinguish)\b',
        r'\b(overview|summary|introduction|intro)\b',
        r'\b(find|discover|locate|identify)\b',
    ],
    QueryIntent.LEARNING: [
        r'\b(learn|learning|study|studying)\b',
        r'\b(tutorial|guide|walkthrough|how.?to)\b',
        r'\b(example|examples?|sample|samples?)\b',
        r'\b(getting started|beginner|introduction|intro)\b',
        r'\b(basics?|fundamental|foundation)\b',
        r'\b(course|lesson|training|workshop)\b',
        r'\b(best practice|pattern|convention)\b',
        r'\b(documentation|docs?|reference|manual)\b',
    ],
    QueryIntent.REFACTORING: [
        r'\b(refactor|refactoring|restructure|reorganize)\b',
        r'\b(improve|enhancement|better|optimize)\b',
        r'\b(clean|cleanup|tidy|simplify)\b',
        r'\b(modernize|upgrade|update|migrate)\b',
        r'\b(reduce|minimize|eliminate|remove)\b',
        r'\b(decouple|separate|extract|split)\b',
        r'\b(consolidate|merge|combine|unify)\b',
        r'\b(rename|move|relocate|reorganize)\b',
        r'\b(dead code|unused|obsolete|deprecated)\b',
        r'\b(code smell|anti.?pattern|technical debt)\b',
    ],
    QueryIntent.PLANNING: [
        r'\b(plan|planning|design|architect)\b',
        r'\b(strategy|strategic|approach|roadmap)\b',
        r'\b(proposal|propose|suggest|recommend)\b',
        r'\b(estimate|estimation|timeline|schedule)\b',
        r'\b(scope|requirement|spec|specification)\b',
        r'\b(milestone|phase|iteration|sprint)\b',
        r'\b(budget|resource|allocation|capacity)\b',
    ],
    QueryIntent.REVIEWING: [
        r'\b(review|reviewing|check|checking)\b',
        r'\b(audit|auditing|inspect|inspection)\b',
        r'\b(validate|validation|verify|verification)\b',
        r'\b(approve|approval|sign.?off)\b',
        r'\b(feedback|comment|suggestion|opinion)\b',
        r'\b(pr|pull request|merge request|code review)\b',
    ],
}

# Domain detection patterns - EXHAUSTIVE (40+ domains, 500+ patterns)
DOMAIN_PATTERNS = {
    # === CORE SOFTWARE ENGINEERING ===
    QueryDomain.SECURITY: [
        r'\b(secur|auth|authent|authoriz)\b',
        r'\b(encrypt|decrypt|cipher|hash|salt|bcrypt|argon)\b',
        r'\b(tokens?|jwt|oauth|oidc|saml|sso|mfa|2fa|totp)\b',
        r'\b(passwords?|credentials?|secrets?|keys?|certs?|certificates?)\b',  # Allow plurals
        r'\b(ssl|tls|https|mtls)\b',
        r'\b(xss|csrf|sqli|injection|sanitiz|escap)\b',
        r'\b(vulnerab|exploit|attack|threat|malicious|breach)\b',
        r'\b(permissions?|roles?|rbac|acl|access control|policy)\b',
        r'\b(owasp|cve|security scan|penetration|pentest)\b',
        r'\b(firewall|waf|ids|ips|siem)\b',
        r'\b(cors|csp|helmet|secure header)\b',
    ],
    QueryDomain.PERFORMANCE: [
        r'\b(perform|fast|slow|quick|speed|latenc)\b',
        r'\b(throughput|bandwidth|capacity|load)\b',
        r'\b(optim|efficien|improv)\b',
        r'\b(cache|cach|redis|memcache|cdn)\b',
        r'\b(memory|heap|stack|gc|garbage collect)\b',
        r'\b(cpu|thread|process|concurren|parallel)\b',
        r'\b(benchmark|profil|flame graph|trace)\b',
        r'\b(bottleneck|hotspot|contention)\b',
        r'\b(scalab|scale|horizontal|vertical)\b',
        r'\b(async|await|non.?block|event loop)\b',
        r'\b(lazy load|prefetch|preload|eager)\b',
        r'\b(compress|gzip|brotli|minif)\b',
        r'\b(pool|connection pool|thread pool)\b',
        r'\b(batch|bulk|chunk|stream)\b',
        r'\b(p50|p90|p95|p99|percentile|sla|slo)\b',
    ],
    QueryDomain.DEBUGGING: [
        r'\b(debug|debugg|debugger)\b',
        r'\b(trace|tracing|traceback)\b',
        r'\b(log|logg|logger|logging)\b',
        r'\b(error|exception|stack trace|stacktrace)\b',
        r'\b(breakpoint|step|watch|inspect)\b',
        r'\b(crash|core dump|segfault|oom)\b',
        r'\b(hang|deadlock|race condition|livelock)\b',
        r'\b(freeze|stuck|unresponsive|timeout)\b',
        r'\b(pdb|gdb|lldb|debugpy|chrome devtools)\b',
        r'\b(print|console|stdout|stderr)\b',
        r'\b(assert|assertion|invariant)\b',
        r'\b(reproduce|repro|minimal|isolate)\b',
        # Vague debugging queries
        r'\b(broke|broken|broke[nd]?|not working|doesnt work|does not work)\b',
        r'\b(wrong|incorrect|unexpected|weird|strange)\b',
        r'\b(fails?|failing|failed)\b',
        r'\b(issue|problem|bug|defect)\b',
    ],
    QueryDomain.ARCHITECTURE: [
        r'\b(architect|design|pattern|struct)\b',
        r'\b(modular|decouple|cohesion|coupling)\b',
        r'\b(layer|tier|hexagonal|onion|clean)\b',
        r'\b(solid|dry|kiss|yagni)\b',
        r'\b(dependency injection|di|ioc|inversion)\b',
        r'\b(interface|abstract|contract|protocol)\b',
        r'\b(factory|singleton|strategy|observer|adapter)\b',
        r'\b(repository|service|controller|handler)\b',
        r'\b(domain driven|ddd|bounded context|aggregate)\b',
        r'\b(cqrs|event sourc|saga|choreograph|orchestrat)\b',
        r'\b(monolith|modular monolith|macro|meso)\b',
        r'\b(component|module|package|namespace)\b',
        r'\b(api gateway|bff|backend for frontend)\b',
    ],
    QueryDomain.TESTING: [
        r'\b(test|testing|tester|tests)\b',
        r'\b(tdd|bdd|atdd)\b',
        r'\b(unit|integration|e2e|end.?to.?end|acceptance)\b',
        r'\b(mock|stub|fake|spy|double)\b',
        r'\b(assert|expect|should|match)\b',
        r'\b(coverage|cov|lcov|istanbul|codecov)\b',
        r'\b(fixture|setup|teardown|before|after)\b',
        r'\b(pytest|jest|junit|mocha|rspec|xunit)\b',
        r'\b(cypress|selenium|playwright|puppeteer)\b',
        r'\b(snapshot|golden|baseline)\b',
        r'\b(parameteriz|data.?driven|table.?driven)\b',
        r'\b(regression|smoke|sanity|load test|stress test)\b',
        r'\b(mutation|property.?based|fuzz)\b',
        r'\b(red.?green|test first|arrange.?act.?assert)\b',
    ],
    QueryDomain.DATABASE: [
        r'\b(database|db|datastore|data store)\b',
        r'\b(sql|nosql|relational|document)\b',
        r'\b(query|select|insert|update|delete|join)\b',
        r'\b(table|column|row|record|field|schema)\b',
        r'\b(index|btree|hash index|composite index)\b',
        r'\b(primary key|foreign key|constraint|unique)\b',
        r'\b(transaction|acid|commit|rollback|isolation)\b',
        r'\b(migration|migrate|flyway|alembic|knex)\b',
        r'\b(postgres|mysql|mariadb|sqlite|oracle|mssql)\b',
        r'\b(mongo|mongodb|dynamodb|cosmos|fauna)\b',
        r'\b(redis|memcached|valkey)\b',
        r'\b(qdrant|pinecone|weaviate|milvus|chroma)\b',
        r'\b(orm|sqlalchemy|typeorm|prisma|sequelize)\b',
        r'\b(pool|connection|replica|shard|partition)\b',
        r'\b(backup|restore|dump|snapshot|point.?in.?time)\b',
        r'\b(deadlock|lock|mutex|optimistic|pessimistic)\b',
        r'\b(stored procedure|trigger|view|materialized)\b',
        r'\b(explain|query plan|analyze|vacuum|reindex)\b',
    ],
    QueryDomain.API: [
        r'\b(api|apis)\b',
        r'\b(rest|restful|http|https)\b',
        r'\b(endpoint|route|path|url|uri)\b',
        r'\b(request|response|payload|body)\b',
        r'\b(get|post|put|patch|delete|head|options)\b',
        r'\b(json|xml|yaml|protobuf|msgpack)\b',
        r'\b(header|cookie|session|bearer)\b',
        r'\b(status code|200|201|400|401|403|404|500)\b',
        r'\b(rate limit|throttl|quota|backoff)\b',
        r'\b(version|v1|v2|deprecat)\b',
        r'\b(swagger|openapi|postman|insomnia)\b',
        r'\b(cors|preflight|origin)\b',
        r'\b(idempoten|retry|timeout|circuit breaker)\b',
        r'\b(pagination|cursor|offset|limit)\b',
        r'\b(hateoas|hypermedia|link)\b',
    ],
    QueryDomain.CONFIGURATION: [
        r'\b(config|configuration|setting|settings)\b',
        r'\b(env|environment|environ|dotenv)\b',
        r'\b(variable|var|param|parameter)\b',
        r'\b(yaml|yml|json|toml|ini|xml)\b',
        r'\b(properties|property|prop)\b',
        r'\b(secrets?|credentials?|sensitive)\b',  # Allow plural
        r'\b(override|default|fallback)\b',
        r'\b(feature flag|toggle|switch)\b',
        r'\b(vault|ssm|secrets? manager|kms)\b',  # Allow plural
        r'\b(12.?factor|twelve.?factor)\b',
    ],
    
    # === DEVOPS & INFRASTRUCTURE ===
    QueryDomain.DEVOPS: [
        r'\b(devops|dev ops|devsecops|sre)\b',
        r'\b(pipeline|workflow|automation)\b',
        r'\b(build|compile|bundle|package)\b',
        r'\b(release|deploy|deployment|rollout)\b',
        r'\b(artifact|registry|repository)\b',
        r'\b(jenkins|gitlab|github actions|azure devops)\b',
        r'\b(ansible|terraform|pulumi|cloudformation)\b',
        r'\b(infrastructure as code|iac)\b',
    ],
    QueryDomain.CI_CD: [
        r'\b(ci|cd|ci.?cd|continuous)\b',
        r'\b(pipeline|workflow|job|stage|step)\b',
        r'\b(build|test|deploy|release)\b',
        r'\b(jenkins|github actions|gitlab ci|circleci|travis)\b',
        r'\b(artifact|cache|artifact cache)\b',
        r'\b(trigger|webhook|schedule|cron)\b',
        r'\b(blue.?green|canary|rolling|feature flag)\b',
        r'\b(approval|gate|manual|automatic)\b',
    ],
    QueryDomain.CONTAINERS: [
        r'\b(container|containers?|containeriz)\b',
        r'\b(docker|dockerfile|docker.?compose|podman)\b',
        r'\b(image|layer|registry|repository)\b',
        r'\b(volume|mount|bind|persist)\b',
        r'\b(network|bridge|overlay|host)\b',
        r'\b(port|expose|publish|map)\b',
        r'\b(entrypoint|cmd|run|exec)\b',
        r'\b(build context|multi.?stage|scratch)\b',
        r'\b(ecr|gcr|acr|dockerhub|ghcr)\b',
    ],
    QueryDomain.KUBERNETES: [
        r'\b(kubernetes|k8s|kube)\b',
        r'\b(pod|deployment|statefulset|daemonset)\b',
        r'\b(service|ingress|loadbalancer|nodeport)\b',
        r'\b(configmap|secret|volume|pvc|pv)\b',
        r'\b(namespace|context|cluster)\b',
        r'\b(kubectl|helm|kustomize|argo|flux)\b',
        r'\b(replica|scale|hpa|vpa|autoscale)\b',
        r'\b(liveness|readiness|probe|health)\b',
        r'\b(istio|linkerd|envoy|service mesh)\b',
        r'\b(operator|crd|custom resource)\b',
        r'\b(eks|gke|aks|openshift|rancher)\b',
    ],
    QueryDomain.CLOUD: [
        r'\b(cloud|aws|azure|gcp|google cloud)\b',
        r'\b(s3|bucket|blob|storage)\b',
        r'\b(ec2|vm|instance|compute)\b',
        r'\b(lambda|function|serverless|faas)\b',
        r'\b(vpc|subnet|security group|network)\b',
        r'\b(iam|role|policy|permission)\b',
        r'\b(rds|aurora|dynamodb|cosmos|firestore)\b',
        r'\b(sns|sqs|eventbridge|pubsub)\b',
        r'\b(cloudwatch|stackdriver|monitor)\b',
        r'\b(cdn|cloudfront|akamai|fastly)\b',
        r'\b(route53|dns|domain|hosted zone)\b',
        r'\b(multi.?region|disaster recovery|dr|backup)\b',
    ],
    QueryDomain.INFRASTRUCTURE: [
        r'\b(infrastructure|infra)\b',
        r'\b(server|host|machine|instance)\b',
        r'\b(network|subnet|vlan|firewall|router)\b',
        r'\b(load balancer|lb|nginx|haproxy|traefik)\b',
        r'\b(dns|domain|nameserver|record)\b',
        r'\b(ssl|tls|certificate|https)\b',
        r'\b(storage|disk|volume|ssd|hdd)\b',
        r'\b(backup|snapshot|replicate|sync)\b',
        r'\b(high availability|ha|failover|redundanc)\b',
        r'\b(proxy|reverse proxy|forward proxy)\b',
    ],
    QueryDomain.MONITORING: [
        r'\b(monitor|monitoring|observ|observabil)\b',
        r'\b(metric|gauge|counter|histogram)\b',
        r'\b(alert|alarm|notification|pager)\b',
        r'\b(prometheus|grafana|datadog|newrelic)\b',
        r'\b(dashboard|visualization|chart|graph)\b',
        r'\b(apm|application performance|trace|span)\b',
        r'\b(health check|heartbeat|uptime|ping)\b',
        r'\b(sla|slo|sli|error budget)\b',
        r'\b(incident|outage|postmortem|runbook)\b',
        r'\b(opentelemetry|otel|jaeger|zipkin)\b',
    ],
    QueryDomain.LOGGING: [
        r'\b(log|logging|logger|logs)\b',
        r'\b(elk|elasticsearch|logstash|kibana)\b',
        r'\b(splunk|sumo|loggly|papertrail)\b',
        r'\b(structured log|json log|format)\b',
        r'\b(level|debug|info|warn|error|fatal)\b',
        r'\b(rotation|retention|archive)\b',
        r'\b(correlation id|trace id|request id)\b',
        r'\b(stdout|stderr|syslog|file)\b',
        r'\b(fluentd|fluent.?bit|filebeat|vector)\b',
    ],
    
    # === FRONTEND ===
    QueryDomain.FRONTEND: [
        r'\b(frontend|front.?end|client.?side|ui|ux)\b',
        r'\b(browser|dom|window|document)\b',
        r'\b(component|widget|element)\b',
        r'\b(state|store|reducer|action)\b',
        r'\b(render|virtual dom|reconciliation)\b',
        r'\b(bundle|webpack|vite|rollup|esbuild)\b',
        r'\b(responsive|mobile|desktop|adaptive)\b',
        r'\b(form|input|validation|submit)\b',
        r'\b(routing|router|navigation|history)\b',
        r'\b(ssr|ssg|csr|hydration|island)\b',
    ],
    QueryDomain.CSS: [
        r'\b(css|style|stylesheet)\b',
        r'\b(sass|scss|less|stylus)\b',
        r'\b(tailwind|bootstrap|material)\b',
        r'\b(flexbox|grid|layout|position)\b',
        r'\b(animation|transition|transform)\b',
        r'\b(responsive|media query|breakpoint)\b',
        r'\b(selector|specificity|cascade)\b',
        r'\b(variable|custom property|theme)\b',
        r'\b(module|scoped|global|import)\b',
        r'\b(styled.?component|emotion|css.?in.?js)\b',
    ],
    QueryDomain.JAVASCRIPT: [
        r'\b(javascript|js|ecmascript|es6|es2020)\b',
        r'\b(typescript|ts|tsx|jsx)\b',
        r'\b(promise|async|await|callback)\b',
        r'\b(closure|scope|hoisting|prototype)\b',
        r'\b(module|import|export|require)\b',
        r'\b(npm|yarn|pnpm|package)\b',
        r'\b(node|deno|bun)\b',
        r'\b(array|object|map|set|weakmap)\b',
        r'\b(event|listener|emit|dispatch)\b',
        r'\b(fetch|axios|xhr|ajax)\b',
    ],
    QueryDomain.REACT: [
        r'\b(react|reactjs|react.?js)\b',
        r'\b(hook|useState|useEffect|useContext|useMemo|useCallback)\b',
        r'\b(component|functional|class component)\b',
        r'\b(props|state|context|ref)\b',
        r'\b(jsx|tsx|render|return)\b',
        r'\b(redux|zustand|recoil|jotai|mobx)\b',
        r'\b(next|nextjs|next.?js|gatsby|remix)\b',
        r'\b(suspense|lazy|concurrent|transition)\b',
        r'\b(react.?query|swr|tanstack)\b',
        r'\b(react.?router|react.?navigation)\b',
    ],
    QueryDomain.VUE: [
        r'\b(vue|vuejs|vue.?js|vue3|vue2)\b',
        r'\b(composition api|options api|setup)\b',
        r'\b(ref|reactive|computed|watch)\b',
        r'\b(component|props|emit|slot)\b',
        r'\b(vuex|pinia|store)\b',
        r'\b(nuxt|nuxtjs|nuxt.?js)\b',
        r'\b(vue.?router|navigation guard)\b',
        r'\b(template|script|style|sfc)\b',
    ],
    QueryDomain.ANGULAR: [
        r'\b(angular|angularjs|ng)\b',
        r'\b(component|directive|pipe|service)\b',
        r'\b(module|ngmodule|standalone)\b',
        r'\b(injectable|provider|dependency injection)\b',
        r'\b(rxjs|observable|subject|subscription)\b',
        r'\b(template|binding|interpolation)\b',
        r'\b(router|route|guard|resolver)\b',
        r'\b(form|reactive form|template driven)\b',
        r'\b(zone|change detection|onpush)\b',
    ],
    QueryDomain.ACCESSIBILITY: [
        r'\b(accessib|a11y|wcag|aria)\b',
        r'\b(screen reader|nvda|voiceover|jaws)\b',
        r'\b(keyboard|focus|tab|navigation)\b',
        r'\b(alt text|label|description)\b',
        r'\b(contrast|color blind|dyslexia)\b',
        r'\b(semantic|landmark|heading)\b',
        r'\b(skip link|focus trap|live region)\b',
    ],
    
    # === BACKEND ===
    QueryDomain.BACKEND: [
        r'\b(backend|back.?end|server.?side)\b',
        r'\b(server|service|api|endpoint)\b',
        r'\b(request|response|handler|controller)\b',
        r'\b(middleware|filter|interceptor)\b',
        r'\b(session|cookie|auth|token)\b',
        r'\b(queue|worker|job|task|scheduler)\b',
        r'\b(cache|redis|memcache)\b',
    ],
    QueryDomain.PYTHON: [
        r'\b(python|py|python3|python2)\b',
        r'\b(pip|pipenv|poetry|conda|venv)\b',
        r'\b(django|flask|fastapi|tornado|starlette)\b',
        r'\b(asyncio|aiohttp|httpx|requests)\b',
        r'\b(pandas|numpy|scipy|matplotlib)\b',
        r'\b(pydantic|dataclass|typing|mypy)\b',
        r'\b(pytest|unittest|nose|coverage)\b',
        r'\b(celery|rq|dramatiq|huey)\b',
        r'\b(sqlalchemy|peewee|tortoise)\b',
        r'\b(__init__|__main__|__name__)\b',
    ],
    QueryDomain.JAVA: [
        r'\b(java|jvm|jdk|jre|javac)\b',
        r'\b(spring|springboot|spring boot|hibernate)\b',
        r'\b(maven|gradle|ant|pom)\b',
        r'\b(bean|annotation|inject|autowire)\b',
        r'\b(servlet|jsp|jpa|jdbc)\b',
        r'\b(stream|lambda|optional|functional)\b',
        r'\b(junit|mockito|testng|assertj)\b',
        r'\b(lombok|guava|jackson|gson)\b',
        r'\b(thread|executor|concurrent|synchronized)\b',
        r'\b(classpath|jar|war|ear)\b',
    ],
    QueryDomain.NODEJS: [
        r'\b(node|nodejs|node.?js)\b',
        r'\b(npm|yarn|pnpm|package.json)\b',
        r'\b(express|koa|fastify|hapi|nest)\b',
        r'\b(middleware|router|controller)\b',
        r'\b(callback|promise|async|await)\b',
        r'\b(stream|buffer|event emitter)\b',
        r'\b(child process|worker thread|cluster)\b',
        r'\b(fs|path|os|http|https)\b',
        r'\b(mocha|jest|ava|tap)\b',
        r'\b(pm2|forever|nodemon)\b',
    ],
    QueryDomain.GOLANG: [
        r'\b(go|golang|gopher)\b',
        r'\b(goroutine|channel|select|defer)\b',
        r'\b(interface|struct|method|receiver)\b',
        r'\b(package|import|module|go.mod)\b',
        r'\b(gin|echo|fiber|chi|mux)\b',
        r'\b(error|panic|recover)\b',
        r'\b(context|cancellation|deadline)\b',
        r'\b(testing|benchmark|go test)\b',
        r'\b(pointer|slice|map|array)\b',
        r'\b(goreleaser|cobra|viper)\b',
    ],
    QueryDomain.RUST: [
        r'\b(rust|rustlang|cargo|crate)\b',
        r'\b(ownership|borrow|lifetime|reference)\b',
        r'\b(trait|impl|struct|enum|match)\b',
        r'\b(option|result|unwrap|expect)\b',
        r'\b(async|await|tokio|async.?std)\b',
        r'\b(actix|axum|rocket|warp)\b',
        r'\b(unsafe|raw pointer|ffi)\b',
        r'\b(macro|derive|attribute)\b',
        r'\b(clippy|rustfmt|rustc)\b',
    ],
    QueryDomain.DOTNET: [
        r'\b(\.?net|dotnet|csharp|c#|asp\.?net)\b',
        r'\b(nuget|package|assembly|dll)\b',
        r'\b(linq|async|await|task)\b',
        r'\b(entity framework|ef|ef core)\b',
        r'\b(mvc|web api|razor|blazor)\b',
        r'\b(dependency injection|di|ioc)\b',
        r'\b(xunit|nunit|mstest|moq)\b',
        r'\b(middleware|pipeline|filter)\b',
        r'\b(visual studio|rider|vs code)\b',
    ],
    
    # === DATA & AI ===
    QueryDomain.DATA_ENGINEERING: [
        r'\b(data engineer|data pipeline|etl|elt)\b',
        r'\b(spark|hadoop|hive|presto|trino)\b',
        r'\b(airflow|dagster|prefect|luigi)\b',
        r'\b(kafka|kinesis|flink|beam)\b',
        r'\b(data lake|data warehouse|lakehouse)\b',
        r'\b(delta|iceberg|hudi|parquet|avro)\b',
        r'\b(dbt|transform|model|materialization)\b',
        r'\b(snowflake|bigquery|redshift|databricks)\b',
    ],
    QueryDomain.MACHINE_LEARNING: [
        r'\b(machine learning|ml|deep learning|dl)\b',
        r'\b(model|train|inference|predict)\b',
        r'\b(neural network|nn|cnn|rnn|transformer)\b',
        r'\b(pytorch|tensorflow|keras|jax)\b',
        r'\b(scikit|sklearn|xgboost|lightgbm)\b',
        r'\b(feature|embedding|vector|tensor)\b',
        r'\b(accuracy|precision|recall|f1|auc)\b',
        r'\b(overfitting|underfitting|regulariz)\b',
        r'\b(hyperparameter|tuning|grid search)\b',
        r'\b(mlflow|wandb|neptune|mlops)\b',
    ],
    QueryDomain.AI: [
        r'\b(ai|artificial intelligence|cognitive)\b',
        r'\b(llm|large language model|gpt|claude|gemini)\b',
        r'\b(rag|retrieval|augmented generation)\b',
        r'\b(embedding|vector|semantic|similarity)\b',
        r'\b(prompt|completion|chat|instruction)\b',
        r'\b(langchain|llamaindex|semantic kernel)\b',
        r'\b(agent|tool|function call|mcp)\b',
        r'\b(fine.?tune|lora|qlora|adapter)\b',
        r'\b(openai|anthropic|huggingface|ollama)\b',
        r'\b(token|context|window|limit)\b',
    ],
    QueryDomain.ANALYTICS: [
        r'\b(analytics|analytic|analysis)\b',
        r'\b(dashboard|report|visualization)\b',
        r'\b(metric|kpi|indicator|measure)\b',
        r'\b(tableau|looker|metabase|superset)\b',
        r'\b(bi|business intelligence)\b',
        r'\b(cohort|funnel|retention|churn)\b',
        r'\b(segment|mixpanel|amplitude|posthog)\b',
    ],
    QueryDomain.ETL: [
        r'\b(etl|elt|extract|transform|load)\b',
        r'\b(data pipeline|data workflow|dag|airflow task)\b',
        r'\b(batch processing|streaming|real.?time data)\b',
        r'\b(data source|data sink|destination|target table)\b',
        r'\b(schema mapping|data mapping|conversion)\b',
        r'\b(data quality|data validation|data profiling)\b',  # Be more specific
        r'\b(incremental load|full load|cdc|change data capture)\b',
    ],
    QueryDomain.DATA_SCIENCE: [
        r'\b(data scien|statistics|statistic)\b',
        r'\b(pandas|numpy|scipy|matplotlib|seaborn)\b',
        r'\b(jupyter|notebook|colab)\b',
        r'\b(regression|classification|clustering)\b',
        r'\b(correlation|distribution|hypothesis)\b',
        r'\b(experiment|ab test|statistical)\b',
        r'\b(feature engineering|preprocessing)\b',
    ],
    
    # === MESSAGING & EVENTS ===
    QueryDomain.MESSAGING: [
        r'\b(messag|queue|broker|pub.?sub)\b',
        r'\b(producer|consumer|subscriber|publisher)\b',
        r'\b(topic|partition|offset|consumer group)\b',
        r'\b(async|asynchronous|event.?driven)\b',
        r'\b(dead letter|dlq|retry|backoff)\b',
        r'\b(exactly.?once|at.?least|at.?most)\b',
    ],
    QueryDomain.KAFKA: [
        r'\b(kafka|confluent|ksql|connect)\b',
        r'\b(topic|partition|replication)\b',
        r'\b(producer|consumer|stream|table)\b',
        r'\b(offset|lag|commit|seek)\b',
        r'\b(avro|schema registry|protobuf)\b',
        r'\b(zookeeper|kraft|broker)\b',
    ],
    QueryDomain.RABBITMQ: [
        r'\b(rabbitmq|rabbit|amqp)\b',
        r'\b(exchange|queue|binding|routing)\b',
        r'\b(direct|fanout|topic|headers)\b',
        r'\b(ack|nack|reject|prefetch)\b',
        r'\b(shovel|federation|cluster)\b',
    ],
    QueryDomain.EVENTS: [
        r'\b(event|events|event.?driven|event.?sourc)\b',
        r'\b(emit|dispatch|subscribe|listen)\b',
        r'\b(handler|listener|callback|hook)\b',
        r'\b(eventbridge|eventgrid|sns|pubsub)\b',
        r'\b(webhook|callback|notification)\b',
        r'\b(saga|choreograph|orchestrat)\b',
    ],
    QueryDomain.STREAMING: [
        r'\b(stream|streaming|real.?time)\b',
        r'\b(flink|spark streaming|kinesis|dataflow)\b',
        r'\b(window|tumbling|sliding|session)\b',
        r'\b(watermark|late|out of order)\b',
        r'\b(state|checkpoint|savepoint)\b',
        r'\b(exactly.?once|at.?least|delivery)\b',
    ],
    
    # === MOBILE ===
    QueryDomain.MOBILE: [
        r'\b(mobile|app|application)\b',
        r'\b(ios|android|cross.?platform)\b',
        r'\b(native|hybrid|pwa)\b',
        r'\b(push notification|notification)\b',
        r'\b(offline|sync|local storage)\b',
        r'\b(gesture|touch|swipe|scroll)\b',
    ],
    QueryDomain.IOS: [
        r'\b(ios|iphone|ipad|apple|macos)\b',
        r'\b(swift|swiftui|uikit|objective.?c)\b',
        r'\b(xcode|simulator|testflight)\b',
        r'\b(cocoapods|spm|swift package)\b',
        r'\b(storyboard|nib|xib|interface builder)\b',
        r'\b(core data|realm|userdefaults)\b',
        r'\b(app store|provisioning|certificate)\b',
    ],
    QueryDomain.ANDROID: [
        r'\b(android|google play|apk|aab)\b',
        r'\b(kotlin|java|jetpack|compose)\b',
        r'\b(activity|fragment|viewmodel|livedata)\b',
        r'\b(gradle|manifest|resource)\b',
        r'\b(room|datastore|sharedpreferences)\b',
        r'\b(material|constraint|recyclerview)\b',
        r'\b(retrofit|okhttp|coroutines|flow)\b',
    ],
    QueryDomain.REACT_NATIVE: [
        r'\b(react native|react.?native|expo)\b',
        r'\b(native module|bridge|turbomodule)\b',
        r'\b(metro|hermes|jsi)\b',
        r'\b(navigation|stack|tab|drawer)\b',
        r'\b(asyncstorage|mmkv)\b',
        r'\b(eas|expo application services)\b',
    ],
    QueryDomain.FLUTTER: [
        r'\b(flutter|dart|widget)\b',
        r'\b(stateless|stateful|buildcontext)\b',
        r'\b(pubspec|pub|package)\b',
        r'\b(bloc|riverpod|provider|getx)\b',
        r'\b(material|cupertino|adaptive)\b',
        r'\b(hot reload|hot restart)\b',
    ],
    
    # === VERSION CONTROL ===
    QueryDomain.GIT: [
        r'\b(git|github|gitlab|bitbucket)\b',
        r'\b(commit|push|pull|fetch|merge)\b',
        r'\b(branch|checkout|rebase|cherry.?pick)\b',
        r'\b(conflict|resolve|diff|patch)\b',
        r'\b(remote|origin|upstream|fork)\b',
        r'\b(stash|reset|revert|amend)\b',
        r'\b(tag|release|version)\b',
        r'\b(hook|pre.?commit|post.?commit)\b',
        r'\b(submodule|subtree|worktree)\b',
        r'\b(gitflow|trunk|feature branch)\b',
    ],
    QueryDomain.VERSION_CONTROL: [
        r'\b(version control|vcs|scm)\b',
        r'\b(commit|revision|changeset)\b',
        r'\b(branch|merge|conflict)\b',
        r'\b(history|blame|annotate)\b',
        r'\b(semver|semantic version)\b',
    ],
    
    # === DOCUMENTATION & COMMUNICATION ===
    QueryDomain.DOCUMENTATION: [
        r'\b(document|documentation|docs?)\b',
        r'\b(readme|changelog|contributing)\b',
        r'\b(comment|docstring|jsdoc|javadoc)\b',
        r'\b(wiki|confluence|notion)\b',
        r'\b(markdown|md|rst|asciidoc)\b',
        r'\b(api doc|swagger|openapi|redoc)\b',
        r'\b(diagram|uml|mermaid|plantuml)\b',
        r'\b(architecture decision|adr)\b',
    ],
    QueryDomain.CODE_REVIEW: [
        r'\b(code review|review|pr|pull request)\b',
        r'\b(comment|feedback|suggestion)\b',
        r'\b(approve|request changes|merge)\b',
        r'\b(diff|change|modification)\b',
        r'\b(lint|format|style|convention)\b',
        r'\b(reviewer|author|contributor)\b',
    ],
    
    # === SPECIFIC TECHNOLOGIES ===
    QueryDomain.GRAPHQL: [
        r'\b(graphql|gql|apollo|relay)\b',
        r'\b(query|mutation|subscription)\b',
        r'\b(schema|type|resolver|directive)\b',
        r'\b(fragment|variable|input)\b',
        r'\b(federation|gateway|subgraph)\b',
        r'\b(n\+1|dataloader|batching)\b',
    ],
    QueryDomain.WEBSOCKET: [
        r'\b(websocket|ws|wss|socket)\b',
        r'\b(real.?time|bidirectional|full.?duplex)\b',
        r'\b(connect|disconnect|message|event)\b',
        r'\b(socket\.?io|ws|sockjs)\b',
        r'\b(ping|pong|heartbeat|keepalive)\b',
        r'\b(room|channel|namespace|broadcast)\b',
    ],
    QueryDomain.GRPC: [
        r'\b(grpc|protobuf|protocol buffer)\b',
        r'\b(service|rpc|method)\b',
        r'\b(unary|streaming|bidirectional)\b',
        r'\b(stub|client|server)\b',
        r'\b(metadata|interceptor|deadline)\b',
        r'\b(proto|message|enum|oneof)\b',
    ],
    QueryDomain.MICROSERVICES: [
        r'\b(microservice|micro.?service)\b',
        r'\b(service mesh|sidecar|proxy)\b',
        r'\b(api gateway|bff)\b',
        r'\b(discovery|registry|consul|eureka)\b',
        r'\b(circuit breaker|retry|timeout)\b',
        r'\b(saga|choreography|orchestration)\b',
        r'\b(distributed|trace|span|correlation)\b',
    ],
    QueryDomain.SERVERLESS: [
        r'\b(serverless|function as service|faas)\b',
        r'\b(lambda|azure function|cloud function)\b',
        r'\b(cold start|warm|invocation)\b',
        r'\b(trigger|event|handler)\b',
        r'\b(timeout|memory|concurrency)\b',
        r'\b(sam|serverless framework|cdk)\b',
        r'\b(step function|workflow|state machine)\b',
    ],
    
    # === QUALITY & COMPLIANCE ===
    QueryDomain.CODE_QUALITY: [
        r'\b(code quality|quality|clean code)\b',
        r'\b(lint|linter|eslint|pylint|rubocop)\b',
        r'\b(format|formatter|prettier|black)\b',
        r'\b(static analysis|sonar|codeclimate)\b',
        r'\b(complexity|cyclomatic|cognitive)\b',
        r'\b(duplication|dry|duplicate)\b',
        r'\b(technical debt|tech debt|refactor)\b',
        r'\b(maintainab|readab|testab)\b',
    ],
    QueryDomain.COMPLIANCE: [
        r'\b(compliance|compliant|regulation)\b',
        r'\b(gdpr|ccpa|hipaa|pci|sox)\b',
        r'\b(audit|audit log|trail)\b',
        r'\b(privacy|consent|data protection)\b',
        r'\b(retention|deletion|anonymiz)\b',
        r'\b(encryption at rest|in transit)\b',
    ],
    QueryDomain.GOVERNANCE: [
        r'\b(governance|policy|standard)\b',
        r'\b(ownership|steward|responsibility)\b',
        r'\b(lifecycle|catalog|lineage)\b',
        r'\b(metadata|classification|tag)\b',
        r'\b(access control|permission|role)\b',
    ],
}

# Domain-specific expansion terms - EXHAUSTIVE (1000+ terms)
DOMAIN_EXPANSIONS = {
    # === CORE SOFTWARE ENGINEERING ===
    QueryDomain.SECURITY: [
        "authentication", "authorization", "encryption", "decryption", "validation",
        "sanitization", "OWASP", "vulnerability", "secure coding", "penetration testing",
        "JWT token", "OAuth 2.0", "OIDC", "SAML", "SSO", "MFA", "2FA", "TOTP",
        "password hashing", "bcrypt", "argon2", "salt", "pepper", "key derivation",
        "SSL/TLS", "HTTPS", "certificate", "mTLS", "public key", "private key",
        "XSS prevention", "CSRF protection", "SQL injection", "input sanitization",
        "CORS", "CSP", "security headers", "helmet", "WAF", "firewall",
        "RBAC", "ACL", "permissions", "roles", "access control", "policy",
        "CVE", "security scan", "SAST", "DAST", "IAST", "dependency audit",
        "secret management", "vault", "KMS", "encryption at rest", "in transit",
    ],
    QueryDomain.PERFORMANCE: [
        "optimization", "caching", "latency", "throughput", "efficiency",
        "profiling", "benchmark", "scalability", "load testing", "stress testing",
        "cache invalidation", "Redis caching", "CDN", "edge caching", "memoization",
        "memory management", "heap", "stack", "garbage collection", "memory leak",
        "CPU utilization", "thread pool", "connection pool", "resource pool",
        "async/await", "non-blocking", "event loop", "concurrency", "parallelism",
        "lazy loading", "eager loading", "prefetch", "preload", "defer",
        "compression", "gzip", "brotli", "minification", "bundling", "tree shaking",
        "database indexing", "query optimization", "N+1 problem", "batch processing",
        "horizontal scaling", "vertical scaling", "auto-scaling", "load balancing",
        "P50", "P90", "P95", "P99", "percentile", "SLA", "SLO", "SLI",
        "flame graph", "trace", "span", "bottleneck", "hotspot", "contention",
    ],
    QueryDomain.DEBUGGING: [
        "troubleshooting", "stack trace", "logging", "breakpoint", "debugging",
        "exception handling", "error message", "root cause analysis", "RCA",
        "print debugging", "console.log", "debugger", "step through", "watch",
        "crash dump", "core dump", "segfault", "memory dump", "heap dump",
        "deadlock", "race condition", "livelock", "thread safety", "synchronization",
        "timeout", "hang", "freeze", "unresponsive", "stuck process",
        "pdb", "gdb", "lldb", "debugpy", "Chrome DevTools", "browser debugging",
        "log levels", "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
        "structured logging", "JSON logs", "correlation ID", "trace ID",
        "reproduce", "minimal reproduction", "isolation", "bisect", "git bisect",
        "assertion", "invariant", "precondition", "postcondition",
    ],
    QueryDomain.ARCHITECTURE: [
        "design pattern", "modularity", "separation of concerns", "SoC",
        "dependency injection", "DI", "IoC", "inversion of control",
        "interface", "abstraction", "contract", "protocol", "API boundary",
        "SOLID principles", "DRY", "KISS", "YAGNI", "composition over inheritance",
        "factory pattern", "singleton", "strategy pattern", "observer pattern",
        "adapter pattern", "decorator pattern", "facade pattern", "proxy pattern",
        "repository pattern", "service layer", "controller", "handler", "use case",
        "domain-driven design", "DDD", "bounded context", "aggregate", "entity",
        "CQRS", "event sourcing", "saga pattern", "choreography", "orchestration",
        "hexagonal architecture", "onion architecture", "clean architecture", "ports and adapters",
        "monolith", "modular monolith", "microservices", "macro services",
        "API gateway", "BFF", "backend for frontend", "anti-corruption layer",
        "layered architecture", "n-tier", "three-tier", "presentation layer",
    ],
    QueryDomain.TESTING: [
        "unit test", "integration test", "e2e test", "end-to-end test", "acceptance test",
        "TDD", "BDD", "ATDD", "test-first", "red-green-refactor",
        "mock", "stub", "fake", "spy", "test double", "mockito", "moq",
        "assertion", "expect", "should", "matcher", "custom matcher",
        "test coverage", "code coverage", "branch coverage", "line coverage",
        "fixture", "setup", "teardown", "beforeEach", "afterEach", "beforeAll",
        "pytest", "jest", "junit", "mocha", "rspec", "xunit", "nunit",
        "Cypress", "Selenium", "Playwright", "Puppeteer", "WebDriver",
        "snapshot testing", "golden test", "visual regression", "screenshot test",
        "parameterized test", "data-driven test", "table-driven test",
        "regression test", "smoke test", "sanity test", "load test", "stress test",
        "mutation testing", "property-based testing", "fuzz testing", "chaos testing",
        "test pyramid", "test isolation", "test parallelization", "flaky test",
    ],
    QueryDomain.DATABASE: [
        "query optimization", "indexing", "B-tree index", "hash index", "composite index",
        "schema design", "normalization", "denormalization", "data modeling",
        "transaction", "ACID", "commit", "rollback", "isolation level",
        "migration", "schema migration", "data migration", "Flyway", "Alembic",
        "PostgreSQL", "MySQL", "MariaDB", "SQLite", "Oracle", "SQL Server",
        "MongoDB", "DynamoDB", "CosmosDB", "Firestore", "Fauna", "CouchDB",
        "Redis", "Memcached", "Valkey", "in-memory database",
        "Qdrant", "Pinecone", "Weaviate", "Milvus", "Chroma", "vector database",
        "ORM", "SQLAlchemy", "TypeORM", "Prisma", "Sequelize", "ActiveRecord",
        "connection pool", "connection string", "replica", "shard", "partition",
        "backup", "restore", "dump", "snapshot", "point-in-time recovery",
        "deadlock", "lock", "mutex", "optimistic locking", "pessimistic locking",
        "stored procedure", "trigger", "view", "materialized view", "CTE",
        "EXPLAIN", "query plan", "ANALYZE", "VACUUM", "REINDEX",
        "primary key", "foreign key", "constraint", "unique constraint",
        "join", "inner join", "outer join", "cross join", "self join",
    ],
    QueryDomain.API: [
        "endpoint", "route", "path", "URL", "URI", "RESTful API",
        "request", "response", "payload", "body", "query parameter",
        "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
        "JSON", "XML", "YAML", "Protocol Buffers", "MessagePack",
        "header", "authorization header", "content-type", "accept",
        "status code", "200 OK", "201 Created", "400 Bad Request", "401 Unauthorized",
        "rate limiting", "throttling", "quota", "backoff", "retry",
        "API versioning", "v1", "v2", "deprecation", "sunset",
        "Swagger", "OpenAPI", "Postman", "Insomnia", "API documentation",
        "CORS", "preflight", "origin", "cross-origin",
        "idempotent", "retry-safe", "timeout", "circuit breaker",
        "pagination", "cursor", "offset", "limit", "page size",
        "HATEOAS", "hypermedia", "link", "rel", "self-describing API",
        "API gateway", "Kong", "Apigee", "AWS API Gateway",
    ],
    QueryDomain.CONFIGURATION: [
        "environment variable", "config file", "settings", "configuration management",
        "dotenv", ".env", "environment", "development", "staging", "production",
        "YAML", "JSON", "TOML", "INI", "XML", "properties file",
        "secret", "credential", "sensitive data", "encryption",
        "override", "default value", "fallback", "hierarchy",
        "feature flag", "toggle", "switch", "LaunchDarkly", "feature management",
        "Vault", "SSM", "Secrets Manager", "KMS", "Azure Key Vault",
        "12-factor app", "config as code", "GitOps", "ConfigMap",
    ],
    
    # === DEVOPS & INFRASTRUCTURE ===
    QueryDomain.DEVOPS: [
        "DevOps", "DevSecOps", "SRE", "site reliability engineering",
        "CI/CD", "continuous integration", "continuous deployment", "continuous delivery",
        "pipeline", "workflow", "automation", "infrastructure as code", "IaC",
        "build", "compile", "bundle", "package", "artifact",
        "release", "deploy", "deployment", "rollout", "rollback",
        "Jenkins", "GitLab CI", "GitHub Actions", "Azure DevOps", "CircleCI",
        "Ansible", "Terraform", "Pulumi", "CloudFormation", "CDK",
        "artifact registry", "container registry", "package repository",
        "blue-green deployment", "canary deployment", "rolling deployment",
        "infrastructure automation", "configuration management", "provisioning",
    ],
    QueryDomain.CI_CD: [
        "continuous integration", "continuous deployment", "CI/CD pipeline",
        "build", "test", "deploy", "release", "artifact",
        "Jenkins", "GitHub Actions", "GitLab CI", "CircleCI", "Travis CI",
        "pipeline stage", "job", "step", "task", "parallel execution",
        "artifact cache", "dependency cache", "build cache",
        "trigger", "webhook", "schedule", "cron", "manual trigger",
        "blue-green", "canary", "rolling", "feature flag",
        "approval gate", "manual approval", "automatic deployment",
        "build matrix", "multi-platform", "cross-compilation",
        "release notes", "changelog", "semantic versioning",
    ],
    QueryDomain.CONTAINERS: [
        "container", "containerization", "Docker", "Podman", "containerd",
        "Dockerfile", "docker-compose", "docker build", "multi-stage build",
        "container image", "layer", "base image", "scratch image",
        "registry", "Docker Hub", "ECR", "GCR", "ACR", "GHCR",
        "volume", "mount", "bind mount", "persistent storage",
        "network", "bridge", "overlay", "host network",
        "port mapping", "expose", "publish", "container port",
        "entrypoint", "CMD", "RUN", "COPY", "ADD",
        "build context", ".dockerignore", "build args",
        "container orchestration", "scaling", "health check",
    ],
    QueryDomain.KUBERNETES: [
        "Kubernetes", "K8s", "container orchestration",
        "Pod", "Deployment", "StatefulSet", "DaemonSet", "ReplicaSet",
        "Service", "Ingress", "LoadBalancer", "NodePort", "ClusterIP",
        "ConfigMap", "Secret", "Volume", "PersistentVolumeClaim", "PVC",
        "namespace", "context", "cluster", "node", "control plane",
        "kubectl", "Helm", "Kustomize", "Argo CD", "Flux",
        "replica", "scaling", "HPA", "VPA", "autoscaling",
        "liveness probe", "readiness probe", "startup probe", "health check",
        "Istio", "Linkerd", "Envoy", "service mesh", "sidecar",
        "operator", "CRD", "custom resource", "controller",
        "EKS", "GKE", "AKS", "OpenShift", "Rancher",
        "resource limits", "requests", "QoS", "priority class",
    ],
    QueryDomain.CLOUD: [
        "cloud computing", "AWS", "Azure", "GCP", "Google Cloud",
        "S3", "bucket", "blob storage", "object storage",
        "EC2", "virtual machine", "instance", "compute",
        "Lambda", "serverless function", "FaaS", "Azure Functions",
        "VPC", "subnet", "security group", "network ACL",
        "IAM", "role", "policy", "permission", "principal",
        "RDS", "Aurora", "DynamoDB", "CosmosDB", "Cloud SQL",
        "SNS", "SQS", "EventBridge", "Pub/Sub", "messaging",
        "CloudWatch", "Stackdriver", "monitoring", "metrics", "logs",
        "CDN", "CloudFront", "Akamai", "Fastly", "edge computing",
        "Route53", "DNS", "domain", "hosted zone", "record",
        "multi-region", "disaster recovery", "DR", "backup", "failover",
        "cost optimization", "reserved instances", "spot instances",
    ],
    QueryDomain.INFRASTRUCTURE: [
        "infrastructure", "server", "host", "machine", "instance",
        "network", "subnet", "VLAN", "firewall", "router", "switch",
        "load balancer", "Nginx", "HAProxy", "Traefik", "ALB", "NLB",
        "DNS", "domain", "nameserver", "A record", "CNAME",
        "SSL/TLS", "certificate", "HTTPS", "Let's Encrypt",
        "storage", "disk", "volume", "SSD", "HDD", "NVMe",
        "backup", "snapshot", "replication", "sync", "DR",
        "high availability", "HA", "failover", "redundancy",
        "proxy", "reverse proxy", "forward proxy", "SOCKS",
        "bastion", "jump host", "VPN", "private network",
    ],
    QueryDomain.MONITORING: [
        # First 3 terms most important - used in query expansion
        "monitoring", "observability", "alert",  # Removed APM - too generic
        "metrics", "gauge", "counter", "histogram", "summary",
        "alerting", "alarm", "notification", "PagerDuty", "OpsGenie",
        "APM", "application performance monitoring",  # Moved down
        "Prometheus", "Grafana", "Datadog", "New Relic", "Dynatrace",
        "dashboard", "visualization", "chart", "graph", "panel",
        "trace", "span", "distributed tracing", "Jaeger", "Zipkin",
        "health check", "heartbeat", "uptime", "ping", "synthetic monitoring",
        "SLA", "SLO", "SLI", "error budget", "availability",
        "incident", "outage", "postmortem", "runbook", "playbook",
        "OpenTelemetry", "OTEL", "collector", "exporter",
    ],
    QueryDomain.LOGGING: [
        "logging", "log management", "log aggregation", "centralized logging",
        "ELK stack", "Elasticsearch", "Logstash", "Kibana",
        "Splunk", "Sumo Logic", "Loggly", "Papertrail",
        "structured logging", "JSON logging", "log format",
        "log level", "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
        "log rotation", "retention", "archive", "compression",
        "correlation ID", "trace ID", "request ID", "span ID",
        "stdout", "stderr", "syslog", "file logging",
        "Fluentd", "Fluent Bit", "Filebeat", "Vector",
        "log search", "query", "filter", "aggregation",
    ],
    
    # === FRONTEND ===
    QueryDomain.FRONTEND: [
        "frontend development", "client-side", "UI", "UX", "user interface",
        "browser", "DOM", "window", "document", "BOM",
        "component", "widget", "element", "prop", "attribute",
        "state management", "store", "reducer", "action", "dispatch",
        "rendering", "virtual DOM", "reconciliation", "diffing",
        "bundler", "Webpack", "Vite", "Rollup", "esbuild", "Parcel",
        "responsive design", "mobile-first", "adaptive design",
        "form", "input", "validation", "submit", "form handling",
        "routing", "router", "navigation", "history API",
        "SSR", "SSG", "CSR", "hydration", "islands architecture",
        "web components", "shadow DOM", "custom elements",
    ],
    QueryDomain.CSS: [
        "CSS", "stylesheet", "styling", "CSS3",
        "Sass", "SCSS", "Less", "Stylus", "PostCSS",
        "Tailwind CSS", "Bootstrap", "Material UI", "Chakra UI",
        "Flexbox", "CSS Grid", "layout", "positioning",
        "animation", "transition", "transform", "keyframes",
        "responsive", "media query", "breakpoint", "viewport",
        "selector", "specificity", "cascade", "inheritance",
        "CSS variable", "custom property", "theming",
        "CSS modules", "scoped CSS", "CSS-in-JS",
        "styled-components", "Emotion", "Linaria",
    ],
    QueryDomain.JAVASCRIPT: [
        "JavaScript", "JS", "ECMAScript", "ES6", "ES2020", "ES2023",
        "TypeScript", "TS", "TSX", "JSX", "type safety",
        "Promise", "async/await", "callback", "event loop",
        "closure", "scope", "hoisting", "prototype", "this",
        "module", "import", "export", "CommonJS", "ESM",
        "npm", "yarn", "pnpm", "package.json", "node_modules",
        "Node.js", "Deno", "Bun", "runtime",
        "Array methods", "Object methods", "Map", "Set", "WeakMap",
        "event handling", "addEventListener", "emit", "dispatch",
        "fetch API", "Axios", "XMLHttpRequest", "AJAX",
    ],
    QueryDomain.REACT: [
        "React", "React.js", "ReactJS", "React 18",
        "hooks", "useState", "useEffect", "useContext", "useMemo", "useCallback",
        "component", "functional component", "class component",
        "props", "state", "context", "ref", "forwardRef",
        "JSX", "TSX", "render", "return", "fragment",
        "Redux", "Zustand", "Recoil", "Jotai", "MobX",
        "Next.js", "Gatsby", "Remix", "React Router",
        "Suspense", "lazy loading", "concurrent mode", "transitions",
        "React Query", "SWR", "TanStack Query", "data fetching",
        "React Testing Library", "Jest", "component testing",
    ],
    QueryDomain.VUE: [
        "Vue.js", "Vue 3", "Vue 2", "Composition API", "Options API",
        "ref", "reactive", "computed", "watch", "watchEffect",
        "component", "props", "emit", "slot", "provide/inject",
        "Vuex", "Pinia", "state management",
        "Nuxt.js", "Nuxt 3", "SSR", "SSG",
        "Vue Router", "navigation guard", "route meta",
        "template", "script setup", "style scoped", "SFC",
        "Vue DevTools", "Vue CLI", "Vite",
    ],
    QueryDomain.ANGULAR: [
        "Angular", "Angular 17", "AngularJS", "Ng",
        "component", "directive", "pipe", "service", "module",
        "NgModule", "standalone component", "lazy loading",
        "dependency injection", "provider", "injectable",
        "RxJS", "Observable", "Subject", "BehaviorSubject", "subscription",
        "template", "binding", "interpolation", "two-way binding",
        "Angular Router", "route guard", "resolver", "lazy loading",
        "reactive forms", "template-driven forms", "form validation",
        "Zone.js", "change detection", "OnPush strategy",
        "Angular CLI", "ng generate", "ng serve", "ng build",
    ],
    QueryDomain.ACCESSIBILITY: [
        "accessibility", "a11y", "WCAG", "ARIA",
        "screen reader", "NVDA", "VoiceOver", "JAWS",
        "keyboard navigation", "focus management", "tab order",
        "alt text", "aria-label", "aria-describedby",
        "color contrast", "color blind", "dyslexia",
        "semantic HTML", "landmark", "heading hierarchy",
        "skip link", "focus trap", "live region",
        "accessible forms", "error announcement", "focus visible",
    ],
    
    # === BACKEND ===
    QueryDomain.BACKEND: [
        "backend development", "server-side", "API development",
        "server", "service", "endpoint", "handler",
        "request handling", "response", "middleware",
        "session management", "cookie", "authentication",
        "background job", "queue", "worker", "scheduler",
        "caching", "Redis", "Memcached", "in-memory cache",
        "database connection", "connection pooling", "ORM",
    ],
    QueryDomain.PYTHON: [
        "Python", "Python 3", "Python 2", "CPython", "PyPy",
        "pip", "pipenv", "Poetry", "conda", "venv", "virtualenv",
        "Django", "Flask", "FastAPI", "Tornado", "Starlette", "ASGI",
        "asyncio", "aiohttp", "httpx", "requests",
        "pandas", "NumPy", "SciPy", "matplotlib", "Jupyter",
        "Pydantic", "dataclass", "typing", "mypy", "type hints",
        "pytest", "unittest", "nose", "coverage", "tox",
        "Celery", "RQ", "Dramatiq", "Huey", "task queue",
        "SQLAlchemy", "Peewee", "Tortoise ORM",
        "__init__.py", "__main__.py", "if __name__ == '__main__'",
    ],
    QueryDomain.JAVA: [
        "Java", "JVM", "JDK", "JRE", "OpenJDK",
        "Spring", "Spring Boot", "Spring Framework", "Hibernate",
        "Maven", "Gradle", "Ant", "pom.xml", "build.gradle",
        "bean", "annotation", "dependency injection", "autowire",
        "servlet", "JSP", "JPA", "JDBC", "entity manager",
        "Stream API", "lambda", "Optional", "functional interface",
        "JUnit", "Mockito", "TestNG", "AssertJ",
        "Lombok", "Guava", "Jackson", "Gson",
        "thread", "executor", "concurrent", "synchronized",
        "classpath", "JAR", "WAR", "EAR", "fat JAR",
    ],
    QueryDomain.NODEJS: [
        "Node.js", "Node", "npm", "yarn", "pnpm",
        "Express.js", "Koa", "Fastify", "Hapi", "NestJS",
        "middleware", "router", "controller", "route handler",
        "callback", "Promise", "async/await", "event loop",
        "stream", "Buffer", "EventEmitter",
        "child_process", "worker_threads", "cluster",
        "fs", "path", "os", "http", "https", "net",
        "Mocha", "Jest", "Ava", "Tap", "Supertest",
        "PM2", "Forever", "nodemon", "process manager",
        "package.json", "node_modules", "npm scripts",
    ],
    QueryDomain.GOLANG: [
        "Go", "Golang", "Gopher",
        "goroutine", "channel", "select", "defer", "panic", "recover",
        "interface", "struct", "method", "receiver", "embedding",
        "package", "import", "module", "go.mod", "go.sum",
        "Gin", "Echo", "Fiber", "Chi", "Gorilla Mux",
        "error handling", "error wrapping", "sentinel error",
        "context", "cancellation", "deadline", "timeout",
        "testing", "benchmark", "go test", "table-driven test",
        "pointer", "slice", "map", "array", "make", "new",
        "GoReleaser", "Cobra", "Viper", "CLI",
    ],
    QueryDomain.RUST: [
        "Rust", "Cargo", "crate", "crates.io",
        "ownership", "borrowing", "lifetime", "reference",
        "trait", "impl", "struct", "enum", "match",
        "Option", "Result", "unwrap", "expect", "?operator",
        "async/await", "Tokio", "async-std", "futures",
        "Actix", "Axum", "Rocket", "Warp",
        "unsafe", "raw pointer", "FFI", "extern",
        "macro", "derive", "attribute", "procedural macro",
        "Clippy", "rustfmt", "rustc", "cargo build",
        "memory safety", "zero-cost abstraction", "fearless concurrency",
    ],
    QueryDomain.DOTNET: [
        ".NET", ".NET Core", "ASP.NET", "C#", "F#", "VB.NET",
        "NuGet", "package", "assembly", "DLL",
        "LINQ", "async/await", "Task", "async programming",
        "Entity Framework", "EF Core", "DbContext",
        "MVC", "Web API", "Razor Pages", "Blazor",
        "dependency injection", "IServiceCollection",
        "xUnit", "NUnit", "MSTest", "Moq",
        "middleware", "pipeline", "filter", "action filter",
        "Visual Studio", "Rider", "VS Code", "dotnet CLI",
    ],
    
    # === DATA & AI ===
    QueryDomain.DATA_ENGINEERING: [
        "data engineering", "data pipeline", "ETL", "ELT",
        "Apache Spark", "Hadoop", "Hive", "Presto", "Trino",
        "Airflow", "Dagster", "Prefect", "Luigi", "workflow orchestration",
        "Kafka", "Kinesis", "Flink", "Apache Beam", "streaming",
        "data lake", "data warehouse", "lakehouse", "data mesh",
        "Delta Lake", "Apache Iceberg", "Hudi", "Parquet", "Avro",
        "dbt", "data transformation", "data modeling",
        "Snowflake", "BigQuery", "Redshift", "Databricks",
        "data quality", "data validation", "data profiling",
    ],
    QueryDomain.MACHINE_LEARNING: [
        "machine learning", "ML", "deep learning", "DL",
        "model training", "inference", "prediction",
        "neural network", "CNN", "RNN", "Transformer", "LSTM",
        "PyTorch", "TensorFlow", "Keras", "JAX",
        "scikit-learn", "XGBoost", "LightGBM", "CatBoost",
        "feature engineering", "feature selection", "embedding",
        "accuracy", "precision", "recall", "F1 score", "AUC-ROC",
        "overfitting", "underfitting", "regularization", "dropout",
        "hyperparameter tuning", "grid search", "random search", "Optuna",
        "MLflow", "Weights & Biases", "Neptune", "MLOps",
    ],
    QueryDomain.AI: [
        "artificial intelligence", "AI", "cognitive computing",
        "LLM", "large language model", "GPT", "Claude", "Gemini", "Llama",
        "RAG", "retrieval-augmented generation", "vector search",
        "embedding", "vector", "semantic similarity", "cosine similarity",
        "prompt engineering", "completion", "chat", "instruction tuning",
        "LangChain", "LlamaIndex", "Semantic Kernel",
        "AI agent", "tool use", "function calling", "MCP",
        "fine-tuning", "LoRA", "QLoRA", "adapter",
        "OpenAI", "Anthropic", "Hugging Face", "Ollama",
        "token", "context window", "token limit", "tokenization",
    ],
    QueryDomain.ANALYTICS: [
        "analytics", "business analytics", "data analytics",
        "dashboard", "report", "visualization", "chart",
        "metrics", "KPI", "key performance indicator",
        "Tableau", "Looker", "Metabase", "Superset",
        "business intelligence", "BI", "self-service analytics",
        "cohort analysis", "funnel analysis", "retention", "churn",
        "Segment", "Mixpanel", "Amplitude", "PostHog",
        "A/B testing", "experimentation", "statistical significance",
    ],
    QueryDomain.ETL: [
        "ETL", "ELT", "extract", "transform", "load",
        "data pipeline", "workflow", "DAG", "task dependency",
        "batch processing", "streaming", "real-time",
        "source", "sink", "destination", "target",
        "schema mapping", "data conversion", "type casting",
        "data quality", "validation", "profiling", "cleansing",
        "incremental load", "full load", "CDC", "change data capture",
        "idempotent", "exactly-once", "at-least-once",
    ],
    QueryDomain.DATA_SCIENCE: [
        "data science", "statistics", "statistical analysis",
        "pandas", "NumPy", "SciPy", "matplotlib", "Seaborn",
        "Jupyter", "notebook", "Google Colab", "interactive computing",
        "regression", "classification", "clustering", "dimensionality reduction",
        "correlation", "distribution", "hypothesis testing",
        "experiment design", "A/B test", "statistical significance",
        "feature engineering", "preprocessing", "normalization",
        "exploratory data analysis", "EDA", "data visualization",
    ],
    
    # === MESSAGING & EVENTS ===
    QueryDomain.MESSAGING: [
        "message queue", "message broker", "pub/sub", "publish-subscribe",
        "producer", "consumer", "subscriber", "publisher",
        "topic", "partition", "offset", "consumer group",
        "asynchronous", "event-driven", "decoupling",
        "dead letter queue", "DLQ", "retry", "backoff",
        "exactly-once", "at-least-once", "at-most-once", "delivery guarantee",
        "message ordering", "FIFO", "message deduplication",
    ],
    QueryDomain.KAFKA: [
        "Apache Kafka", "Confluent", "KSQL", "Kafka Connect",
        "topic", "partition", "replication factor",
        "producer", "consumer", "Kafka Streams", "KTable",
        "offset", "lag", "commit", "seek", "rebalance",
        "Avro", "Schema Registry", "Protobuf", "JSON Schema",
        "ZooKeeper", "KRaft", "broker", "controller",
        "consumer group", "partition assignment", "offset management",
    ],
    QueryDomain.RABBITMQ: [
        "RabbitMQ", "AMQP", "message broker",
        "exchange", "queue", "binding", "routing key",
        "direct exchange", "fanout exchange", "topic exchange", "headers exchange",
        "ack", "nack", "reject", "prefetch",
        "shovel", "federation", "cluster", "high availability",
        "dead letter exchange", "TTL", "message expiration",
    ],
    QueryDomain.EVENTS: [
        "event", "event-driven architecture", "event sourcing",
        "emit", "dispatch", "subscribe", "listen",
        "event handler", "event listener", "callback", "hook",
        "EventBridge", "Event Grid", "SNS", "Cloud Pub/Sub",
        "webhook", "callback URL", "notification",
        "saga pattern", "choreography", "orchestration",
        "event store", "event replay", "event projection",
    ],
    QueryDomain.STREAMING: [
        "stream processing", "streaming", "real-time",
        "Apache Flink", "Spark Streaming", "Kinesis", "Dataflow",
        "windowing", "tumbling window", "sliding window", "session window",
        "watermark", "late data", "out-of-order", "event time",
        "state management", "checkpoint", "savepoint",
        "exactly-once", "at-least-once", "delivery semantics",
        "backpressure", "buffer", "throughput",
    ],
    
    # === MOBILE ===
    QueryDomain.MOBILE: [
        "mobile development", "mobile app", "application",
        "iOS", "Android", "cross-platform",
        "native", "hybrid", "PWA", "progressive web app",
        "push notification", "local notification", "FCM", "APNs",
        "offline", "sync", "local storage", "SQLite",
        "gesture", "touch", "swipe", "scroll", "pinch",
        "app store", "Google Play", "distribution",
    ],
    QueryDomain.IOS: [
        "iOS", "iPhone", "iPad", "Apple", "macOS",
        "Swift", "SwiftUI", "UIKit", "Objective-C",
        "Xcode", "Simulator", "TestFlight", "App Store Connect",
        "CocoaPods", "Swift Package Manager", "Carthage",
        "Storyboard", "NIB", "XIB", "Interface Builder",
        "Core Data", "Realm", "UserDefaults", "Keychain",
        "App Store", "provisioning profile", "certificate", "signing",
    ],
    QueryDomain.ANDROID: [
        "Android", "Google Play", "APK", "AAB",
        "Kotlin", "Java", "Jetpack", "Compose",
        "Activity", "Fragment", "ViewModel", "LiveData",
        "Gradle", "AndroidManifest", "resource", "layout",
        "Room", "DataStore", "SharedPreferences",
        "Material Design", "ConstraintLayout", "RecyclerView",
        "Retrofit", "OkHttp", "Coroutines", "Flow",
    ],
    QueryDomain.REACT_NATIVE: [
        "React Native", "Expo", "Metro bundler",
        "native module", "bridge", "TurboModule", "JSI",
        "Hermes", "JavaScript Core", "native code",
        "React Navigation", "stack", "tab", "drawer",
        "AsyncStorage", "MMKV", "persistent storage",
        "EAS", "Expo Application Services", "OTA update",
        "platform-specific", "iOS", "Android", "native code",
    ],
    QueryDomain.FLUTTER: [
        "Flutter", "Dart", "widget",
        "StatelessWidget", "StatefulWidget", "BuildContext",
        "pubspec.yaml", "pub", "package",
        "BLoC", "Riverpod", "Provider", "GetX",
        "Material", "Cupertino", "adaptive design",
        "hot reload", "hot restart", "development cycle",
        "platform channel", "method channel", "native plugin",
    ],
    
    # === VERSION CONTROL ===
    QueryDomain.GIT: [
        "Git", "GitHub", "GitLab", "Bitbucket",
        "commit", "push", "pull", "fetch", "merge",
        "branch", "checkout", "rebase", "cherry-pick",
        "conflict", "resolve", "diff", "patch",
        "remote", "origin", "upstream", "fork",
        "stash", "reset", "revert", "amend",
        "tag", "release", "version",
        "hook", "pre-commit", "post-commit", "pre-push",
        "submodule", "subtree", "worktree",
        "GitFlow", "trunk-based", "feature branch",
    ],
    QueryDomain.VERSION_CONTROL: [
        "version control", "VCS", "SCM",
        "commit", "revision", "changeset", "history",
        "branch", "merge", "conflict resolution",
        "blame", "annotate", "bisect",
        "semantic versioning", "semver", "version bump",
    ],
    
    # === DOCUMENTATION & COMMUNICATION ===
    QueryDomain.DOCUMENTATION: [
        "documentation", "docs", "technical writing",
        "README", "CHANGELOG", "CONTRIBUTING", "LICENSE",
        "comment", "docstring", "JSDoc", "Javadoc", "TSDoc",
        "wiki", "Confluence", "Notion", "knowledge base",
        "Markdown", "reStructuredText", "AsciiDoc",
        "API documentation", "Swagger", "OpenAPI", "Redoc",
        "diagram", "UML", "Mermaid", "PlantUML", "draw.io",
        "architecture decision record", "ADR", "RFC",
    ],
    QueryDomain.CODE_REVIEW: [
        "code review", "PR review", "pull request",
        "comment", "feedback", "suggestion", "improvement",
        "approve", "request changes", "merge",
        "diff", "change", "modification", "addition", "deletion",
        "lint", "format", "style", "convention",
        "reviewer", "author", "contributor", "assignee",
        "code quality", "best practice", "anti-pattern",
    ],
    
    # === SPECIFIC TECHNOLOGIES ===
    QueryDomain.GRAPHQL: [
        "GraphQL", "GQL", "Apollo", "Relay",
        "query", "mutation", "subscription",
        "schema", "type", "resolver", "directive",
        "fragment", "variable", "input type",
        "federation", "gateway", "subgraph",
        "N+1 problem", "DataLoader", "batching",
        "introspection", "schema stitching", "persisted query",
    ],
    QueryDomain.WEBSOCKET: [
        "WebSocket", "WS", "WSS", "socket",
        "real-time", "bidirectional", "full-duplex",
        "connect", "disconnect", "message", "event",
        "Socket.IO", "ws", "SockJS",
        "ping", "pong", "heartbeat", "keepalive",
        "room", "channel", "namespace", "broadcast",
        "connection state", "reconnection", "backoff",
    ],
    QueryDomain.GRPC: [
        "gRPC", "Protocol Buffers", "Protobuf",
        "service", "RPC", "method", "procedure",
        "unary", "streaming", "bidirectional streaming",
        "stub", "client", "server", "channel",
        "metadata", "interceptor", "deadline", "timeout",
        "proto file", "message", "enum", "oneof",
        "code generation", "grpc-web", "transcoding",
    ],
    QueryDomain.MICROSERVICES: [
        "microservices", "micro-services", "distributed system",
        "service mesh", "sidecar", "proxy", "Envoy",
        "API gateway", "BFF", "backend for frontend",
        "service discovery", "registry", "Consul", "Eureka",
        "circuit breaker", "retry", "timeout", "bulkhead",
        "saga", "choreography", "orchestration",
        "distributed tracing", "span", "trace", "correlation ID",
        "eventual consistency", "CAP theorem", "BASE",
    ],
    QueryDomain.SERVERLESS: [
        "serverless", "function as a service", "FaaS",
        "Lambda", "Azure Functions", "Cloud Functions", "Edge Functions",
        "cold start", "warm start", "invocation",
        "trigger", "event", "handler", "context",
        "timeout", "memory", "concurrency", "provisioned",
        "SAM", "Serverless Framework", "CDK", "Pulumi",
        "Step Functions", "workflow", "state machine",
        "event-driven", "pay-per-use", "auto-scaling",
    ],
    
    # === QUALITY & COMPLIANCE ===
    QueryDomain.CODE_QUALITY: [
        "code quality", "clean code", "maintainability",
        "linter", "ESLint", "Pylint", "RuboCop", "golint",
        "formatter", "Prettier", "Black", "gofmt",
        "static analysis", "SonarQube", "CodeClimate",
        "cyclomatic complexity", "cognitive complexity",
        "code duplication", "DRY", "copy-paste detection",
        "technical debt", "tech debt", "refactoring",
        "maintainability index", "readability", "testability",
    ],
    QueryDomain.COMPLIANCE: [
        "compliance", "regulation", "regulatory",
        "GDPR", "CCPA", "HIPAA", "PCI-DSS", "SOX", "SOC 2",
        "audit", "audit log", "audit trail",
        "privacy", "consent", "data protection", "data subject rights",
        "retention", "deletion", "anonymization", "pseudonymization",
        "encryption at rest", "encryption in transit",
        "access control", "least privilege", "need to know",
    ],
    QueryDomain.GOVERNANCE: [
        "governance", "policy", "standard", "guideline",
        "data ownership", "data steward", "responsibility",
        "data lifecycle", "data catalog", "data lineage",
        "metadata management", "classification", "tagging",
        "access control", "permission", "role-based",
        "data quality", "data integrity", "data accuracy",
    ],
    
    # === GENERAL (catch-all expansions) ===
    QueryDomain.GENERAL: [
        "software development", "programming", "coding",
        "best practice", "pattern", "anti-pattern",
        "design", "implementation", "architecture",
        "debug", "troubleshoot", "fix", "resolve",
        "optimize", "improve", "enhance", "refactor",
        "test", "verify", "validate", "quality",
        "documentation", "comment", "readme",
        "deploy", "release", "production", "staging",
    ],
}

# Intent-specific prompt templates
ANALYTICAL_TEMPLATE = """
OBJECTIVE: Analyze and understand {topic}

CONTEXT: {context}

ANALYSIS FOCUS:
- {focus_points}

Provide insights about:
- Key patterns and relationships
- Gaps or discrepancies
- Strategic implications
"""

IMPLEMENTATION_TEMPLATE = """
OBJECTIVE: {action} {target}

REQUIREMENTS:
- {requirements}

APPROACH:
- {approach}

CONSTRAINTS:
- Follow existing patterns
- Maintain backward compatibility
"""

TROUBLESHOOTING_TEMPLATE = """
OBJECTIVE: Diagnose and resolve {issue}

SYMPTOMS: {symptoms}

INVESTIGATION AREAS:
- {investigation_points}

Expected outcome: {expected}
"""


class StructuredQueryEnhancer:
    """Enhance queries using .enhancedprompt.md methodology."""
    
    def __init__(self):
        """Initialize the enhancer."""
        pass
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the query intent."""
        query_lower = query.lower()
        
        # Check each intent pattern
        intent_scores = {}
        for intent, patterns in INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower, re.IGNORECASE))
            intent_scores[intent] = score
        
        # Return highest scoring intent, default to EXPLORATORY
        max_intent = max(intent_scores, key=intent_scores.get)
        return max_intent if intent_scores[max_intent] > 0 else QueryIntent.EXPLORATORY
    
    def detect_domains(self, query: str) -> List[QueryDomain]:
        """Detect relevant technical domains."""
        query_lower = query.lower()
        domains = []
        
        for domain, patterns in DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    domains.append(domain)
                    break
        
        return domains if domains else [QueryDomain.GENERAL]
    
    def expand_query(self, query: str, domains: List[QueryDomain]) -> Tuple[str, List[str]]:
        """Expand query with domain-specific terminology."""
        expansions = []
        
        for domain in domains:
            if domain in DOMAIN_EXPANSIONS:
                # Add first 5 expansion terms per domain for better coverage
                expansions.extend(DOMAIN_EXPANSIONS[domain][:5])
        
        # Combine original query with expansions
        if expansions:
            enhanced = f"{query} {' '.join(expansions)}"
        else:
            enhanced = query
        
        return enhanced, expansions
    
    def create_structured_prompt(
        self, query: str, intent: QueryIntent, domains: List[QueryDomain]
    ) -> str:
        """Create a structured prompt based on intent and domain."""
        
        if intent == QueryIntent.ANALYTICAL:
            return f"""Analyze: {query}
Focus areas: {', '.join(d.value for d in domains)}
Provide insights on patterns, relationships, and strategic implications."""
        
        elif intent == QueryIntent.IMPLEMENTATION:
            return f"""Implement: {query}
Domains: {', '.join(d.value for d in domains)}
Follow best practices and existing patterns."""
        
        elif intent == QueryIntent.TROUBLESHOOTING:
            return f"""Debug: {query}
Investigation areas: {', '.join(d.value for d in domains)}
Identify root cause and provide resolution."""
        
        else:  # EXPLORATORY
            return f"""Explain: {query}
Related topics: {', '.join(d.value for d in domains)}
Provide clear, actionable information."""
    
    def enhance(self, query: str) -> EnhancedQuery:
        """
        Enhance a query using the full .enhancedprompt.md methodology.
        
        Args:
            query: Original user query
        
        Returns:
            EnhancedQuery with all enhancements
        """
        # Step 1: Classify intent
        intent = self.classify_intent(query)
        
        # Step 2: Detect domains
        domains = self.detect_domains(query)
        
        # Step 3: Expand query with domain terminology
        enhanced_query, expansion_terms = self.expand_query(query, domains)
        
        # Step 4: Create structured prompt
        structured_prompt = self.create_structured_prompt(query, intent, domains)
        
        return EnhancedQuery(
            original_query=query,
            intent=intent,
            domains=domains,
            enhanced_query=enhanced_query,
            expansion_terms=expansion_terms,
            structured_prompt=structured_prompt,
        )


def test_structured_enhancer():
    """Test the structured query enhancer."""
    enhancer = StructuredQueryEnhancer()
    
    print("=" * 80)
    print("STRUCTURED QUERY ENHANCEMENT TEST (.enhancedprompt.md methodology)")
    print("=" * 80)
    
    test_queries = [
        "fix database errors",
        "compare microservices vs monolith",
        "how to implement rate limiting",
        "slow API response times",
        "understand authentication flow",
        "broken tests after refactor",
        "add caching to database queries",
        "what is the best security practice",
    ]
    
    for query in test_queries:
        result = enhancer.enhance(query)
        print(f"\nOriginal: \"{query}\"")
        print(f"Intent: {result.intent.value}")
        print(f"Domains: {[d.value for d in result.domains]}")
        print(f"Expansions: {result.expansion_terms}")
        print(f"Enhanced: \"{result.enhanced_query}\"")
        print("-" * 40)
        print(f"Structured Prompt:\n{result.structured_prompt}")
    
    print("\n" + "=" * 80)
    print("STRUCTURED ENHANCEMENT READY FOR USE")
    print("=" * 80)


if __name__ == "__main__":
    test_structured_enhancer()
