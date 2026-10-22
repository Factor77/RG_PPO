"""AUST rolling PPO-only runner with the original AUST environment semantics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.agents.stablebaselines3.models import DRLEnsembleAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split


class AUSTPPOOnlyAgent(DRLEnsembleAgent):
    """Run AUST's rolling workflow with PPO as its sole candidate model.

    The parent class still provides the original AUST PPO training, validation,
    trading, turbulence, and previous-state hand-off behavior. The only removed
    operations are training and selecting non-PPO candidate models.
    """

    def run_ppo_only_strategy(self, ppo_model_kwargs, timesteps_dict):
        """Execute AUST rolling PPO training without ensemble selection."""

        last_state_ensemble = []
        ppo_sharpe_list = []
        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[
            (self.df.date < self.train_period[1])
            & (self.df.date >= self.train_period[0])
        ]
        insample_turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.90
        )

        start = time.time()
        for i in range(
            self.rebalance_window + self.validation_window,
            len(self.unique_trade_date),
            self.rebalance_window,
        ):
            validation_start_date = self.unique_trade_date[
                i - self.rebalance_window - self.validation_window
            ]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]
            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            initial = i == self.rebalance_window + self.validation_window

            end_date_index = self.df.index[
                self.df["date"] == validation_start_date
            ].to_list()[-1]
            historical_turbulence = self.df.iloc[
                end_date_index - 63 + 1 : end_date_index + 1, :
            ].drop_duplicates(subset=["date"])
            historical_turbulence_mean = np.mean(
                historical_turbulence.turbulence.values
            )
            if historical_turbulence_mean > insample_turbulence_threshold:
                turbulence_threshold = insample_turbulence_threshold
            else:
                turbulence_threshold = np.quantile(
                    insample_turbulence.turbulence.values, 1
                )

            # This is the effective threshold in FinRL's AUST ensemble runner.
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
            print("turbulence_threshold: ", turbulence_threshold)

            train = data_split(self.df, self.train_period[0], validation_start_date)
            self.train_env = DummyVecEnv(
                [
                    lambda: StockTradingEnv(
                        df=train,
                        stock_dim=self.stock_dim,
                        hmax=self.hmax,
                        initial_amount=self.initial_amount,
                        num_stock_shares=[0] * self.stock_dim,
                        buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                        sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                        reward_scaling=self.reward_scaling,
                        state_space=self.state_space,
                        action_space=self.action_space,
                        tech_indicator_list=self.tech_indicator_list,
                        print_verbosity=self.print_verbosity,
                    )
                ]
            )
            validation = data_split(self.df, validation_start_date, validation_end_date)

            model_ppo, ppo_sharpe_list, _ = self._train_window(
                "ppo",
                ppo_model_kwargs,
                ppo_sharpe_list,
                validation_start_date,
                validation_end_date,
                timesteps_dict,
                i,
                validation,
                turbulence_threshold,
            )
            if model_ppo is None:
                raise RuntimeError("PPO training was disabled or failed")

            model_use.append("PPO")
            last_state_ensemble = self.DRL_prediction(
                model=model_ppo,
                name="ppo",
                last_state=last_state_ensemble,
                iter_num=i,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
            )

        print("AUST PPO-only strategy took: ", (time.time() - start) / 60, " minutes")
        return pd.DataFrame(
            {
                "Iter": iteration_list,
                "Val Start": validation_start_date_list,
                "Val End": validation_end_date_list,
                "Model Used": model_use,
                "PPO Sharpe": ppo_sharpe_list,
            }
        )
