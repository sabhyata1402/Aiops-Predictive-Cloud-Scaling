/**
 * load_test.js — k6 performance load test for the AIOps metrics endpoint.
 *
 * Simulates a realistic ramp-up → peak → ramp-down load profile.
 * Run while watching the Live Monitor tab to see alerts trigger in real time.
 *
 * Prerequisites:
 *   brew install k6          (macOS)
 *   sudo apt install k6      (Ubuntu/Debian)
 *
 * Usage:
 *   k6 run src/monitoring/load_test.js
 *
 * To target your GCP VM instead of localhost, set the BASE_URL env var:
 *   k6 run -e BASE_URL=http://<VM_IP>:8080 src/monitoring/load_test.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

const errorRate = new Rate("errors");
const BASE_URL  = __ENV.BASE_URL || "http://localhost:8080";

export const options = {
  stages: [
    { duration: "30s", target: 10  },  // ramp-up:   0  → 10  VUs
    { duration: "1m",  target: 50  },  // ramp-up:   10 → 50  VUs (load peak)
    { duration: "2m",  target: 50  },  // hold peak: sustained high load
    { duration: "30s", target: 100 },  // spike:     50 → 100 VUs (SLA breach zone)
    { duration: "1m",  target: 50  },  // recover:   100 → 50
    { duration: "30s", target: 0   },  // ramp-down: 50 → 0
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],  // 95% of requests under 500ms
    errors:            ["rate<0.05"],  // error rate under 5%
  },
};

export default function () {
  // Hit the metrics endpoint (simulates monitoring scrape)
  const metricsRes = http.get(`${BASE_URL}/metrics`);
  const metricsOk  = check(metricsRes, {
    "metrics status 200": (r) => r.status === 200,
    "metrics has cpu":    (r) => JSON.parse(r.body).cpu !== undefined,
  });
  errorRate.add(!metricsOk);

  // Hit health endpoint
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, { "health status 200": (r) => r.status === 200 });

  sleep(0.1);  // 100ms think time → ~10 req/s per VU
}
