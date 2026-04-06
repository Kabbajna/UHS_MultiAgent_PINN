#!/usr/bin/env python3
"""
MARL Extension for Multi-Agent UHS Orchestrator
================================================

4 modifications on top of the existing orchestrator:
1. Gated Communication (H → G → Y → H)
2. Hidden State Summaries for Consensus (14 → 110 dim)
3. Per-Agent MARL with physics rewards (Phase 2)
4. Phase 3 (Communication) + Phase 4 (Recalibration)

Constraints:
- Phase 1a/1b identical to original
- MC Dropout works (communication is deterministic)
- Agent architectures unchanged (modules added around, not inside)
- Physical veto unchanged
- Backward compatible: disable communication → original behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

from orchestrator import (
    PhysicsInformedHydroAgent, GeochemAgent, HysteresisAgent,
    UncertaintyAgent, AttentionReasoningLayer, PINNMultiAgentOrchestrator,
    engineer_hydro_physics_features, physics_loss_hydro,
    train_pinn_hydro, train_agent, imitation_learning, evaluate,
    DEVICE
)


# =============================================================================
# 1. HIDDEN STATE EXTRACTION (added methods — no architecture change)
# =============================================================================

def _hydro_forward_with_hidden(self, x):
    """Run Hydro backbone, return output + hidden state (256-dim)."""
    h = self.input_proj(x)
    for block in self.res_blocks:
        h = block(h)
    h = self.attention(h)
    hidden = h
    physics_out = self.physics_branch(h)
    data_out = self.data_branch(h)
    weights = F.softmax(self.branch_weights, dim=0)
    output = weights[0] * physics_out + weights[1] * data_out
    return output, hidden


def _hydro_output_from_hidden(self, h):
    """Re-run only the dual-branch output head on a (possibly modified) hidden state."""
    physics_out = self.physics_branch(h)
    data_out = self.data_branch(h)
    weights = F.softmax(self.branch_weights, dim=0)
    return weights[0] * physics_out + weights[1] * data_out


def _uncertainty_forward_with_hidden(self, x):
    """Run UncertaintyAgent backbone, return output + hidden state (hidden-dim)."""
    layers = list(self.net.children())
    h = x
    for layer in layers[:-1]:  # all except final Linear
        h = layer(h)
    hidden = h
    output = layers[-1](h)
    return output, hidden


def _uncertainty_output_from_hidden(self, h):
    """Re-run only the output layer on a (possibly modified) hidden state."""
    return list(self.net.children())[-1](h)


# Monkey-patch the methods onto agent classes
PhysicsInformedHydroAgent.forward_with_hidden = _hydro_forward_with_hidden
PhysicsInformedHydroAgent.output_from_hidden = _hydro_output_from_hidden
UncertaintyAgent.forward_with_hidden = _uncertainty_forward_with_hidden
UncertaintyAgent.output_from_hidden = _uncertainty_output_from_hidden


# =============================================================================
# 2. GATED MESSAGE MODULE
# =============================================================================

class GatedMessage(nn.Module):
    """
    Single directed message channel with sigmoid gate.

    sender_hidden → project → message (d_m)
    receiver_hidden + message → gate → gated_message
    [receiver_hidden; gated_message] → reduce → updated_hidden
    """
    def __init__(self, sender_dim, receiver_dim, message_dim=32):
        super().__init__()
        self.project = nn.Linear(sender_dim, message_dim)
        self.gate = nn.Linear(receiver_dim + message_dim, message_dim)
        self.reduce = nn.Sequential(
            nn.Linear(receiver_dim + message_dim, receiver_dim),
            nn.SiLU(),
        )

    def forward(self, h_sender, h_receiver):
        """
        Args:
            h_sender: [batch, sender_dim] — sender's hidden state
            h_receiver: [batch, receiver_dim] — receiver's hidden state
        Returns:
            h_receiver_updated: [batch, receiver_dim]
        """
        message = self.project(h_sender)                          # [batch, d_m]
        gate_input = torch.cat([h_receiver, message], dim=-1)     # [batch, recv + d_m]
        gate_weights = torch.sigmoid(self.gate(gate_input))       # [batch, d_m]
        gated_message = gate_weights * message                    # element-wise
        combined = torch.cat([h_receiver, gated_message], dim=-1) # [batch, recv + d_m]
        return self.reduce(combined)                              # [batch, recv_dim]


# =============================================================================
# 3. INTER-AGENT COMMUNICATION MODULE
# =============================================================================

class InterAgentCommunication(nn.Module):
    """
    Gated communication cycle: H → G → Y → H.

    Deterministic (no dropout) — safe for MC Dropout inference.
    """
    def __init__(self, hydro_hidden=256, geochem_hidden=128, hyst_hidden=128, message_dim=32):
        super().__init__()
        self.h_to_g = GatedMessage(hydro_hidden, geochem_hidden, message_dim)
        self.g_to_y = GatedMessage(geochem_hidden, hyst_hidden, message_dim)
        self.y_to_h = GatedMessage(hyst_hidden, hydro_hidden, message_dim)

    def forward(self, h_hydro, h_geochem, h_hyst):
        """
        Two-pass message cycle:
        Pass 1: H→G, G→Y (sequential)
        Pass 2: Y→H (closes the loop)

        Returns updated hidden states for all three agents.
        """
        # Pass 1: H → G → Y
        h_geochem_updated = self.h_to_g(h_hydro, h_geochem)
        h_hyst_updated = self.g_to_y(h_geochem_updated, h_hyst)

        # Pass 2: Y → H (feedback)
        h_hydro_updated = self.y_to_h(h_hyst_updated, h_hydro)

        return h_hydro_updated, h_geochem_updated, h_hyst_updated

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# 4. HIDDEN STATE SUMMARY MODULE
# =============================================================================

class HiddenStateSummary(nn.Module):
    """
    Project each agent's hidden state to a summary vector (ℝ^32).
    Concatenated summaries (96-dim) are added to consensus input.
    """
    def __init__(self, hydro_hidden=256, geochem_hidden=128, hyst_hidden=128, summary_dim=32):
        super().__init__()
        self.hydro_proj = nn.Linear(hydro_hidden, summary_dim)
        self.geochem_proj = nn.Linear(geochem_hidden, summary_dim)
        self.hyst_proj = nn.Linear(hyst_hidden, summary_dim)
        self.summary_dim = summary_dim

    def forward(self, h_hydro, h_geochem, h_hyst):
        """Returns concatenated summary [batch, 96]."""
        s_h = self.hydro_proj(h_hydro)
        s_g = self.geochem_proj(h_geochem)
        s_y = self.hyst_proj(h_hyst)
        return torch.cat([s_h, s_g, s_y], dim=-1)  # [batch, 96]


# =============================================================================
# 5. STOCHASTIC AGENT WRAPPER (for REINFORCE)
# =============================================================================

class StochasticAgentWrapper(nn.Module):
    """
    Wraps a frozen/unfrozen agent with a learnable log_std for REINFORCE.
    Does NOT modify the agent architecture.
    """
    def __init__(self, agent, output_dim):
        super().__init__()
        self.agent = agent
        self.log_std = nn.Parameter(torch.ones(output_dim) * -2)

    def forward(self, x):
        return self.agent(x)

    def sample(self, x):
        mean = self.agent(x)
        std = torch.exp(self.log_std.clamp(-5, 0))
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(-1)
        return sample, log_prob, mean

    def sample_from_hidden(self, h):
        """Sample using output_from_hidden (for Hydro re-execution after message)."""
        mean = self.agent.output_from_hidden(h)
        std = torch.exp(self.log_std.clamp(-5, 0))
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(-1)
        return sample, log_prob, mean


# =============================================================================
# 6. MARL ORCHESTRATOR
# =============================================================================

class MARLOrchestrator(nn.Module):
    """
    Extended orchestrator with gated communication and hidden state summaries.

    Toggle `use_communication` for ablations:
    - False: identical to original PINNMultiAgentOrchestrator
    - True: communication + summaries active
    """
    def __init__(self, hydro_agent, geochem_agent, hysteresis_agent,
                 hidden=256, message_dim=32, summary_dim=32):
        super().__init__()

        self.hydro = hydro_agent
        self.geochem = geochem_agent
        self.hysteresis = hysteresis_agent

        # Communication module (deterministic — no dropout)
        self.communication = InterAgentCommunication(
            hydro_hidden=256, geochem_hidden=128, hyst_hidden=128,
            message_dim=message_dim
        )

        # Hidden state summaries
        self.summary = HiddenStateSummary(
            hydro_hidden=256, geochem_hidden=128, hyst_hidden=128,
            summary_dim=summary_dim
        )

        # Original consensus layer (input_dim=14)
        self.reasoning = AttentionReasoningLayer(
            input_dim=14, output_dim=3, n_agents=3, hidden=hidden
        )

        # Summary injection: additive projection into consensus hidden space
        # Zero-initialized so it has NO effect until trained
        self.summary_inject = nn.Linear(summary_dim * 3, hidden)
        nn.init.zeros_(self.summary_inject.weight)
        nn.init.zeros_(self.summary_inject.bias)

        # Toggle
        self.use_communication = False

        # Stochastic wrappers for MARL (created lazily)
        self._stochastic_wrappers = None

    def _get_stochastic_wrappers(self):
        if self._stochastic_wrappers is None:
            self._stochastic_wrappers = {
                'hydro': StochasticAgentWrapper(self.hydro, 3).to(next(self.parameters()).device),
                'geochem': StochasticAgentWrapper(self.geochem, 5).to(next(self.parameters()).device),
                'hysteresis': StochasticAgentWrapper(self.hysteresis, 2).to(next(self.parameters()).device),
            }
        return self._stochastic_wrappers

    def freeze_agents(self):
        """Freeze all agent parameters."""
        for agent in [self.hydro, self.geochem, self.hysteresis]:
            for p in agent.parameters():
                p.requires_grad = False

    def unfreeze_agents(self):
        """Unfreeze all agent parameters."""
        for agent in [self.hydro, self.geochem, self.hysteresis]:
            for p in agent.parameters():
                p.requires_grad = True

    def freeze_communication(self):
        for p in self.communication.parameters():
            p.requires_grad = False
        for p in self.summary.parameters():
            p.requires_grad = False
        for p in self.summary_inject.parameters():
            p.requires_grad = False

    def unfreeze_communication(self):
        for p in self.communication.parameters():
            p.requires_grad = True
        for p in self.summary.parameters():
            p.requires_grad = True
        for p in self.summary_inject.parameters():
            p.requires_grad = True

    def freeze_consensus(self):
        for p in self.reasoning.parameters():
            p.requires_grad = False

    def unfreeze_consensus(self):
        for p in self.reasoning.parameters():
            p.requires_grad = True

    def forward(self, x_enriched, x_hydro_enriched, deterministic=True, return_details=False):
        """
        Forward pass with optional communication.

        When use_communication=False: identical to original orchestrator.
        When use_communication=True: gated message cycle + hidden summaries.
        """
        x_flow = x_enriched[:, 0:4]
        x_geochem = x_enriched[:, 4:12]
        x_hyst = x_enriched[:, 12:15]

        if not self.use_communication:
            # === ORIGINAL BEHAVIOR (no communication) ===
            with torch.no_grad() if not any(p.requires_grad for p in self.hydro.parameters()) else torch.enable_grad():
                hydro_pred = self.hydro(x_hydro_enriched)
                geochem_pred = self.geochem(x_geochem)
                hyst_pred = self.hysteresis(x_hyst)

            combined = torch.cat([x_flow, hydro_pred, geochem_pred, hyst_pred], dim=-1)
            agent_outputs = [hydro_pred, geochem_pred, hyst_pred]

            if deterministic:
                output = self.reasoning(combined, agent_outputs)
                log_prob = None
            else:
                output, log_prob, _ = self.reasoning.sample(combined, agent_outputs)

        else:
            # === COMMUNICATION-ENABLED BEHAVIOR ===
            # Step 1: Get hidden states from all agents
            grad_ctx = torch.enable_grad() if any(
                p.requires_grad for p in self.hydro.parameters()
            ) else torch.no_grad()
            with grad_ctx:
                hydro_pred_init, h_hydro = self.hydro.forward_with_hidden(x_hydro_enriched)
                geochem_pred_init, h_geochem = self.geochem.forward_with_hidden(x_geochem)
                hyst_pred_init, h_hyst = self.hysteresis.forward_with_hidden(x_hyst)

            # Step 2: Gated communication cycle (H→G→Y→H)
            h_hydro_upd, h_geochem_upd, h_hyst_upd = self.communication(
                h_hydro, h_geochem, h_hyst
            )

            # Step 3: Re-execute output layers on updated hidden states
            with grad_ctx:
                hydro_pred = self.hydro.output_from_hidden(h_hydro_upd)
                geochem_pred = self.geochem.output_from_hidden(h_geochem_upd)
                hyst_pred = self.hysteresis.output_from_hidden(h_hyst_upd)

            # Step 4: Compute hidden state summaries
            summaries = self.summary(h_hydro_upd, h_geochem_upd, h_hyst_upd)  # [batch, 96]

            # Step 5: Consensus with summary injection
            combined = torch.cat([x_flow, hydro_pred, geochem_pred, hyst_pred], dim=-1)
            agent_outputs = [hydro_pred, geochem_pred, hyst_pred]

            if deterministic:
                output = self._forward_with_summary(combined, agent_outputs, summaries)
                log_prob = None
            else:
                output, log_prob = self._sample_with_summary(combined, agent_outputs, summaries)

        if return_details:
            attention = self.reasoning.get_attention_weights()
            return output, log_prob, {
                'hydro': hydro_pred, 'geochem': geochem_pred,
                'hysteresis': hyst_pred, 'attention': attention,
            }
        return output, log_prob

    def _forward_with_summary(self, combined, agent_outputs, summaries):
        """Forward through reasoning with additive summary injection."""
        # Run input projection
        h = self.reasoning.input_proj(combined)
        # Inject summary (additive — zero-init means no effect initially)
        h = h + self.summary_inject(summaries)
        # Continue with encoder + attention + decoder
        features = h + self.reasoning.encoder(h)
        query = self.reasoning.attention_query(features)

        agent_values, agent_keys = [], []
        for i, (proj, agent_out) in enumerate(zip(self.reasoning.agent_projections, agent_outputs)):
            padded = F.pad(agent_out, (0, features.size(-1) - agent_out.size(-1)))
            projected = proj(padded)
            agent_values.append(self.reasoning.attention_value(projected))
            agent_keys.append(self.reasoning.attention_key(projected))

        keys = torch.stack(agent_keys, dim=1)
        values = torch.stack(agent_values, dim=1)
        query = query.unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(-1, -2)) / np.sqrt(keys.size(-1))
        attention_weights = F.softmax(scores, dim=-1)
        self.reasoning.last_attention_weights = attention_weights.squeeze(1).detach()
        attended = torch.bmm(attention_weights, values).squeeze(1)
        combined_feat = features + attended
        decoded = combined_feat + self.reasoning.decoder(combined_feat)
        return self.reasoning.mean_head(decoded)

    def _sample_with_summary(self, combined, agent_outputs, summaries):
        """Stochastic sampling with summary injection."""
        mean = self._forward_with_summary(combined, agent_outputs, summaries)
        std = torch.exp(self.reasoning.log_std.clamp(-5, 0))
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).sum(-1)
        return sample, log_prob

    def predict_with_uncertainty(self, x_enriched, x_hydro_enriched, n_samples=100):
        """MC Dropout inference (communication is deterministic)."""
        self.train()  # Activate dropout in agents + consensus
        predictions, attentions = [], []

        for _ in range(n_samples):
            pred, _, details = self.forward(
                x_enriched, x_hydro_enriched, deterministic=True, return_details=True
            )
            predictions.append(pred)
            attentions.append(details['attention'])

        self.eval()
        preds = torch.stack(predictions)
        return preds.mean(dim=0), preds.std(dim=0), torch.stack(attentions).mean(dim=0)


# =============================================================================
# 7. PHASE 2: PER-AGENT MARL FINE-TUNING
# =============================================================================

def marl_finetuning(orchestrator, X_calib, X_hydro_calib, Y_calib,
                    X_mrst, X_mrst_physics, Y_mrst, scaler_Y_mrst,
                    X_phreeqc, Y_phreeqc, scaler_Y_geochem,
                    X_bc, Y_bc, scaler_Y_bc,
                    scaler_Y, gamma=0.3, epochs=100, lr=1e-5):
    """
    Phase 2: Per-agent MARL with physics rewards.

    All agents + consensus unfrozen. Communication stays frozen (not yet trained).
    Each agent gets its own physics reward + γ × collaborative reward.
    """
    orchestrator = orchestrator.to(DEVICE)
    orchestrator.use_communication = False  # Communication not active in Phase 2

    # Unfreeze agents + consensus
    orchestrator.unfreeze_agents()
    orchestrator.unfreeze_consensus()
    orchestrator.freeze_communication()

    # Separate optimizers per agent + consensus (for gradient normalization)
    opt_hydro = optim.Adam(orchestrator.hydro.parameters(), lr=lr)
    opt_geochem = optim.Adam(orchestrator.geochem.parameters(), lr=lr)
    opt_hyst = optim.Adam(orchestrator.hysteresis.parameters(), lr=lr)
    opt_consensus = optim.Adam(orchestrator.reasoning.parameters(), lr=lr)

    # Stochastic wrappers
    wrappers = orchestrator._get_stochastic_wrappers()

    # Move data to device
    X_t = torch.tensor(X_calib, dtype=torch.float32).to(DEVICE)
    X_h = torch.tensor(X_hydro_calib, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(Y_calib, dtype=torch.float32).to(DEVICE)

    X_mrst_t = torch.tensor(X_mrst_physics, dtype=torch.float32).to(DEVICE)
    Y_mrst_t = torch.tensor(Y_mrst, dtype=torch.float32).to(DEVICE)

    X_phreeqc_t = torch.tensor(X_phreeqc, dtype=torch.float32).to(DEVICE)
    Y_phreeqc_t = torch.tensor(Y_phreeqc, dtype=torch.float32).to(DEVICE)

    X_bc_t = torch.tensor(X_bc, dtype=torch.float32).to(DEVICE)
    Y_bc_t = torch.tensor(Y_bc, dtype=torch.float32).to(DEVICE)

    Y_mean = torch.tensor(scaler_Y.mean_, dtype=torch.float32).to(DEVICE)
    Y_std = torch.tensor(scaler_Y.scale_, dtype=torch.float32).to(DEVICE)

    beta = 0.1  # Supervised auxiliary loss coefficient
    batch_size = 256

    print(f"\nPhase 2: Per-Agent MARL ({epochs} epochs, γ={gamma})...")

    for epoch in range(epochs):
        epoch_rewards = {'hydro': [], 'geochem': [], 'hyst': [], 'collab': []}

        # --- A) Per-agent updates on decoupled data ---

        # Hydro on MRST
        perm = torch.randperm(len(X_mrst_t))[:batch_size]
        x_batch, y_batch = X_mrst_t[perm], Y_mrst_t[perm]
        opt_hydro.zero_grad()
        pred_h, lp_h, _ = wrappers['hydro'].sample(x_batch)
        err_h = (pred_h - y_batch).pow(2).mean(-1)
        pred_orig = pred_h * Y_std + Y_mean
        sat_err = F.relu(pred_orig[:, 1] + pred_orig[:, 2] - 1.0)
        r_hydro = torch.exp(-err_h) * 5.0 + torch.exp(-5 * sat_err)
        adv_h = (r_hydro - r_hydro.mean()) / (r_hydro.std() + 1e-8)
        loss_h = -(lp_h * adv_h.detach()).mean() + beta * err_h.mean()
        if not torch.isnan(loss_h):
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(orchestrator.hydro.parameters(), 0.5)
            opt_hydro.step()
        epoch_rewards['hydro'].append(r_hydro.mean().item())

        # Geochem on PHREEQC
        perm = torch.randperm(len(X_phreeqc_t))[:batch_size]
        x_batch, y_batch = X_phreeqc_t[perm], Y_phreeqc_t[perm]
        opt_geochem.zero_grad()
        pred_g, lp_g, _ = wrappers['geochem'].sample(x_batch)
        err_g = (pred_g - y_batch).pow(2).mean(-1)
        r_geochem = torch.exp(-err_g) * 5.0
        adv_g = (r_geochem - r_geochem.mean()) / (r_geochem.std() + 1e-8)
        loss_g = -(lp_g * adv_g.detach()).mean() + beta * err_g.mean()
        if not torch.isnan(loss_g):
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(orchestrator.geochem.parameters(), 0.5)
            opt_geochem.step()
        epoch_rewards['geochem'].append(r_geochem.mean().item())

        # Hysteresis on Brooks-Corey
        perm = torch.randperm(len(X_bc_t))[:batch_size]
        x_batch, y_batch = X_bc_t[perm], Y_bc_t[perm]
        opt_hyst.zero_grad()
        pred_y, lp_y, _ = wrappers['hysteresis'].sample(x_batch)
        err_y = (pred_y - y_batch).pow(2).mean(-1)
        r_hyst = torch.exp(-err_y) * 5.0
        adv_y = (r_hyst - r_hyst.mean()) / (r_hyst.std() + 1e-8)
        loss_y = -(lp_y * adv_y.detach()).mean() + beta * err_y.mean()
        if not torch.isnan(loss_y):
            loss_y.backward()
            torch.nn.utils.clip_grad_norm_(orchestrator.hysteresis.parameters(), 0.5)
            opt_hyst.step()
        epoch_rewards['hyst'].append(r_hyst.mean().item())

        # --- B) Collaborative update on coupled data ---
        perm = torch.randperm(len(X_t))[:batch_size]
        batch_x, batch_h, batch_y = X_t[perm], X_h[perm], Y_t[perm]

        # Zero all optimizers
        opt_hydro.zero_grad()
        opt_geochem.zero_grad()
        opt_hyst.zero_grad()
        opt_consensus.zero_grad()

        pred, log_prob = orchestrator(batch_x, batch_h, deterministic=False)

        # Collaborative reward
        data_error = (pred - batch_y).pow(2).mean(-1)
        data_reward = torch.exp(-data_error)
        pred_orig = pred * Y_std + Y_mean
        sat_error = (pred_orig[:, 1] + pred_orig[:, 2] - 1.0).abs()
        sat_reward = torch.exp(-5 * sat_error)
        r_collab = data_reward * 5.0 + sat_reward

        baseline = r_collab.mean()
        advantage = (r_collab - baseline) / (r_collab.std() + 1e-8)
        policy_loss = -(log_prob * advantage.detach()).mean()
        supervised_loss = data_error.mean() * beta
        loss_collab = gamma * (policy_loss + supervised_loss)

        if not torch.isnan(loss_collab):
            loss_collab.backward()
            # Normalize gradients per agent before stepping
            torch.nn.utils.clip_grad_norm_(orchestrator.hydro.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(orchestrator.geochem.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(orchestrator.hysteresis.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(orchestrator.reasoning.parameters(), 0.5)
            opt_hydro.step()
            opt_geochem.step()
            opt_hyst.step()
            opt_consensus.step()

        epoch_rewards['collab'].append(r_collab.mean().item())

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: R_H={np.mean(epoch_rewards['hydro']):.3f} "
                  f"R_G={np.mean(epoch_rewards['geochem']):.3f} "
                  f"R_Y={np.mean(epoch_rewards['hyst']):.3f} "
                  f"R_collab={np.mean(epoch_rewards['collab']):.3f}")

    # Re-freeze agents after MARL
    orchestrator.freeze_agents()
    return orchestrator


# =============================================================================
# 8. PHASE 3: COMMUNICATION FINE-TUNING
# =============================================================================

def train_communication(orchestrator, X_calib, X_hydro_calib, Y_calib, epochs=100, lr=1e-4):
    """
    Phase 3: Train communication + consensus jointly.

    Agents frozen. Communication + summary + consensus unfrozen.
    End-to-end MSE on coupled data.
    """
    orchestrator = orchestrator.to(DEVICE)
    orchestrator.use_communication = True

    # Freeze agents only, unfreeze communication + consensus
    orchestrator.freeze_agents()
    orchestrator.unfreeze_consensus()
    orchestrator.unfreeze_communication()

    # Optimizer for communication + summary + consensus
    comm_params = (
        list(orchestrator.communication.parameters()) +
        list(orchestrator.summary.parameters()) +
        list(orchestrator.summary_inject.parameters()) +
        list(orchestrator.reasoning.parameters())
    )
    optimizer = optim.Adam(comm_params, lr=lr)

    X_t = torch.tensor(X_calib, dtype=torch.float32).to(DEVICE)
    X_h = torch.tensor(X_hydro_calib, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(Y_calib, dtype=torch.float32).to(DEVICE)

    criterion = nn.MSELoss()
    batch_size = 256

    print(f"\nPhase 3: Communication fine-tuning ({epochs} epochs)...")
    print(f"  Trainable params: {sum(p.numel() for p in comm_params):,}")

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        orchestrator.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = orchestrator(X_t[idx], X_h[idx], deterministic=True)
            loss = criterion(pred, Y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comm_params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in orchestrator.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    if best_state:
        orchestrator.load_state_dict(best_state)
    print(f"  Best Loss: {best_loss:.6f}")
    return orchestrator


# =============================================================================
# 9. PHASE 4: RECALIBRATION
# =============================================================================

def recalibrate(orchestrator, X_calib, X_hydro_calib, Y_calib, epochs=200, lr=1e-4):
    """
    Phase 4: Joint fine-tuning of consensus + communication.

    Agents frozen. Consensus + communication unfrozen.
    Allows consensus to adapt to communication-modified inputs.
    """
    orchestrator = orchestrator.to(DEVICE)
    orchestrator.use_communication = True

    # Freeze agents, unfreeze consensus + communication
    orchestrator.freeze_agents()
    orchestrator.unfreeze_consensus()
    orchestrator.unfreeze_communication()

    trainable_params = (
        list(orchestrator.reasoning.parameters()) +
        list(orchestrator.communication.parameters()) +
        list(orchestrator.summary.parameters()) +
        list(orchestrator.summary_inject.parameters())
    )
    optimizer = optim.Adam(trainable_params, lr=lr)

    X_t = torch.tensor(X_calib, dtype=torch.float32).to(DEVICE)
    X_h = torch.tensor(X_hydro_calib, dtype=torch.float32).to(DEVICE)
    Y_t = torch.tensor(Y_calib, dtype=torch.float32).to(DEVICE)

    criterion = nn.MSELoss()
    batch_size = 256

    print(f"\nPhase 4: Recalibration ({epochs} epochs, lr={lr})...")

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        orchestrator.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = orchestrator(X_t[idx], X_h[idx], deterministic=True)
            loss = criterion(pred, Y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.3)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in orchestrator.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    if best_state:
        orchestrator.load_state_dict(best_state)
    print(f"  Best Loss: {best_loss:.6f}")
    return orchestrator


# =============================================================================
# 10. EVALUATION WITH/WITHOUT COMMUNICATION (for ablations)
# =============================================================================

def evaluate_marl(orchestrator, X_test, X_hydro_test, Y_test, scaler_Y,
                  use_communication=True):
    """Evaluate with toggle for communication ablation."""
    prev = orchestrator.use_communication
    orchestrator.use_communication = use_communication
    orchestrator.eval()

    batch_size = 5000
    all_preds = []

    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        X_batch = torch.tensor(X_test[i:end], dtype=torch.float32).to(DEVICE)
        X_h_batch = torch.tensor(X_hydro_test[i:end], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred, _ = orchestrator(X_batch, X_h_batch, deterministic=True)
        all_preds.append(pred.cpu().numpy())

    pred_np = np.vstack(all_preds)
    pred_orig = scaler_Y.inverse_transform(pred_np)

    r2 = r2_score(Y_test, pred_orig, multioutput='uniform_average')
    r2_P = r2_score(Y_test[:, 0], pred_orig[:, 0])
    r2_Sw = r2_score(Y_test[:, 1], pred_orig[:, 1])
    r2_Sg = r2_score(Y_test[:, 2], pred_orig[:, 2])

    orchestrator.use_communication = prev
    return {'r2': r2, 'r2_P': r2_P, 'r2_Sw': r2_Sw, 'r2_Sg': r2_Sg,
            'communication': use_communication}
