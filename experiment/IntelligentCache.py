import torch
from utils.inference_utils import (
    infer_single_sample,
    calculate_confidence_intervals
)
from utils.log_utils import debug


class IntelligentCache:

    def __init__(self, redis_client, model, device, config_settings, mc_samples=10):
        self.redis = redis_client
        self.model = model
        self.device = device
        self.config_settings = config_settings
        self.mc_samples = mc_samples


    def ttl_dynamic(self, p, conf):
        alpha, beta, base = (
            self.config_settings.ttl_alpha,
            self.config_settings.ttl_beta,
            self.config_settings.ttl_base
        )
        return base * (1 + alpha * p) * (1 + beta * conf)


    def should_prefetch(self, p, conf):
        return (
            p > self.config_settings.p_threshold
            and conf > self.config_settings.conf_threshold
        )


    def should_evict(self, p, conf):
        return (
            p < self.config_settings.p_threshold
            or conf < self.config_settings.conf_threshold
        )


    def serve_request(self, request_key, features):
        key_str = str(request_key)

        try:
            # prepare input
            x_tensor = (torch.tensor(features).unsqueeze(0)
                        .float().to(self.device))
            y_dummy = (torch.zeros((1,), dtype=torch.long).
                       to(self.device))

            # inference
            outputs_mean, outputs_var = infer_single_sample(
                self.model,
                x_tensor,
                [key_str],
                y_dummy,
                self.device,
                mc_samples=self.mc_samples
            )

            # calculate CIs
            lower_ci, upper_ci = calculate_confidence_intervals(
                [outputs_mean.cpu()],
                [outputs_var.cpu()],
                self.config_settings
            )

            # softmax applied on boundaries
            lower_probs = torch.softmax(
                lower_ci.squeeze(0),
                dim=0
            )
            upper_probs = torch.softmax(
                upper_ci.squeeze(0),
                dim=0
            )

            # boundaries for the predicted class
            p_lower = lower_probs[1].item()
            p_upper = upper_probs[1].item()

            # debugging
            debug(f"🔎 CIs: [{p_lower}, {p_upper}]")

            #
            if p_lower < self.config_settings.p_threshold:
                self.redis.delete(key_str)
                return None, "evicted"

            # Cache HIT
            cached_value = self.redis.get(key_str)
            if cached_value:
                return cached_value.decode(), "hit"

            # TTL dinamico basato su CI
            ttl = self.ttl_dynamic(p_lower, p_upper - p_lower)

            # Cache MISS
            value = f"value_for_{key_str}"
            self.redis.set(key_str, value, ex=int(ttl))

            # PREFETCH solo se siamo molto sicuri (p_lower alto)
            if p_lower > self.config_settings.p_threshold:
                for offset in range(1, 3):
                    next_key = str(int(request_key) + offset)
                    if not self.redis.exists(next_key):
                        self.redis.set(next_key, f"prefetch_{next_key}", ex=int(ttl))

            return value, "miss"

        except Exception as e:
            raise RuntimeError(f"❌ Failed to serve request '{key_str}': {e}")