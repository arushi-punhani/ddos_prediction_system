import uuid
from locust import HttpUser, task, between, constant

PRODUCT_QUERY = """
{
  products(first: 5) {
    edges {
      node {
        id
        name
      }
    }
  }
}
"""

HEAVY_QUERY = """
{
  products(first: 20) {
    edges {
      node {
        id
        name
      }
    }
  }
}
"""


class LegitimateCustomer(HttpUser):
    # Realistic human browsing pace: read, scroll, decide, then act again.
    wait_time = between(1.5, 6)

    def on_start(self):
        self.client_id = str(uuid.uuid4())

    @task
    def browse(self):
        self.client.post(
            "/graphql/",
            json={"query": PRODUCT_QUERY},
            headers={"X-Client-Id": self.client_id},
            name="Normal_Browse"
        )


class WarningUser(HttpUser):
    wait_time = between(0.2, 0.5)

    def on_start(self):
        self.client_id = str(uuid.uuid4())

    @task
    def browse(self):
        self.client.post(
            "/graphql/",
            json={"query": HEAVY_QUERY},
            headers={"X-Client-Id": self.client_id},
            name="Warning_Sim"
        )


class HighVolumeLegitimate(HttpUser):
    """
    Simulates a legitimate traffic spike (e.g. flash sale, marketing push).
    Many distinct real users, each browsing normally but a bit more
    urgently/frequently than usual. Key difference from DDoSStressAttacker:
    requests are spread across MANY distinct clients at a moderate,
    still-paced per-client rate, not concentrated in a few clients
    hammering at wait_time=0.
    """
    # Faster/more impulsive than baseline browsing (sale urgency),
    # but still clearly paced -- not zero-wait like the attacker.
    wait_time = between(0.8, 2.5)

    def on_start(self):
        self.client_id = str(uuid.uuid4())

    @task
    def browse(self):
        self.client.post(
            "/graphql/",
            json={"query": PRODUCT_QUERY},
            headers={"X-Client-Id": self.client_id},
            name="HighVolume_Legit"
        )


class DDoSStressAttacker(HttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.client_id = str(uuid.uuid4())

    @task
    def flood(self):
        self.client.post(
            "/graphql/",
            json={"query": PRODUCT_QUERY},
            headers={"X-Client-Id": self.client_id},
            name="Malicious_Flood"
        )