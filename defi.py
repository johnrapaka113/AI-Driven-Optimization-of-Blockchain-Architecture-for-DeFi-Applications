import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sqlite3
import bcrypt
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from streamlit_autorefresh import st_autorefresh
from prophet import Prophet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from transformers.pipelines import Pipeline
except ImportError:
    from transformers import pipeline as Pipeline
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import logging
import ta
import shap
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import torch_geometric
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Streamlit app configuration
st.set_page_config(page_title="AI Blockchain Finance Platform", layout="wide")

# CoinGecko API setup
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Initialize databases
def init_db():
    try:
        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/portfolio.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS portfolios (username TEXT, coin_id TEXT, amount REAL, PRIMARY KEY (username, coin_id))''')
        conn.commit()
        conn.close()
        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
        conn.commit()
        conn.close()
        logger.info("Databases initialized")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        st.error(f"Database error: {str(e)}")

init_db()

# Authentication functions
def authenticate(username, password):
    try:
        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
            logger.info(f"User {username} authenticated")
            return True
        logger.warning(f"Authentication failed for {username}")
        return False
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        st.error(f"Authentication error: {str(e)}")
        return False

def register_user(username, password):
    try:
        if not username or not password:
            raise ValueError("Username and password cannot be empty")
        conn = sqlite3.connect('data/users.db')
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            raise ValueError("Username already exists")
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        logger.info(f"User {username} registered")
    except Exception as e:
        logger.error(f"Registration error: {e}")
        st.error(f"Registration error: {str(e)}")
        raise

# Portfolio functions
def save_portfolio(username, portfolio):
    try:
        conn = sqlite3.connect('data/portfolio.db')
        c = conn.cursor()
        c.execute("DELETE FROM portfolios WHERE username = ?", (username,))
        for coin_id, amount in portfolio.items():
            if amount > 0:
                c.execute("INSERT INTO portfolios (username, coin_id, amount) VALUES (?, ?, ?)", (username, coin_id, amount))
        conn.commit()
        conn.close()
        logger.info(f"Portfolio saved for {username}")
    except Exception as e:
        logger.error(f"Portfolio save error: {e}")
        st.error(f"Portfolio save error: {str(e)}")

def load_portfolio(username):
    try:
        conn = sqlite3.connect('data/portfolio.db')
        c = conn.cursor()
        c.execute("SELECT coin_id, amount FROM portfolios WHERE username = ?", (username,))
        result = c.fetchall()
        conn.close()
        logger.info(f"Portfolio loaded for {username}")
        return {coin_id: amount for coin_id, amount in result}
    except Exception as e:
        logger.error(f"Portfolio load error: {e}")
        st.error(f"Portfolio load error: {str(e)}")
        return {}

# CoinGecko data fetching
@st.cache_data(ttl=60)
def fetch_coin_list():
    try:
        response = session.get(f"{COINGECKO_BASE_URL}/coins/markets?vs_currency=usd&per_page=250&category=decentralized-finance-defi", timeout=5)
        coins = response.json()
        dex_coins = ["bitcoin", "ethereum", "uniswap", "pancakeswap", "sushi", "curve-dao-token"]
        logger.info("Fetched coin list")
        return {coin['name']: coin['id'] for coin in coins if coin['id'] in dex_coins or coin['market_cap_rank'] is not None}
    except Exception as e:
        logger.error(f"Coin list fetch error: {e}")
        return {"Bitcoin": "bitcoin", "Ethereum": "ethereum", "Uniswap": "uniswap", "PancakeSwap": "pancakeswap"}

@st.cache_data(ttl=30)
def fetch_price(coin_id):
    try:
        response = session.get(f"{COINGECKO_BASE_URL}/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true", timeout=5)
        response.raise_for_status()
        data = response.json()[coin_id]
        logger.info(f"Fetched price for {coin_id}")
        return data['usd'], data['usd_24h_change']
    except Exception as e:
        logger.error(f"Price fetch error for {coin_id}: {e}")
        return 100, 0

@st.cache_data(ttl=3600)
def fetch_bulk_prices(coin_ids):
    try:
        ids = ','.join(coin_ids)
        response = session.get(f"{COINGECKO_BASE_URL}/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true", timeout=5)
        response.raise_for_status()
        data = response.json()
        logger.info("Fetched bulk prices")
        return {coin_id: (data[coin_id]['usd'], data[coin_id]['usd_24h_change']) for coin_id in coin_ids}
    except Exception as e:
        logger.error(f"Bulk prices fetch error: {e}")
        return {coin_id: (100, 0) for coin_id in coin_ids}

@st.cache_data(ttl=3600)
def fetch_historical_data(coin_id, days):
    try:
        days = min(days, 90)
        response = session.get(f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily", timeout=5)
        data = response.json()
        prices = [entry[1] for entry in data['prices']]
        volumes = [entry[1] for entry in data['total_volumes']]
        changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 if i > 0 else 0 for i in range(len(prices))]
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'change_24h': changes,
            'rsi': ta.momentum.RSIIndicator(pd.Series(prices)).rsi(),
            'macd': ta.trend.MACD(pd.Series(prices)).macd(),
            'volatility': pd.Series(prices).rolling(window=14).std(),
            'ds': pd.date_range(end=pd.Timestamp.now(), periods=len(prices), freq='D')
        }).fillna(method='bfill')
        if len(df) > 50:
            df = df.iloc[::len(df)//50]
        logger.info(f"Fetched historical data for {coin_id}")
        return df
    except Exception as e:
        logger.error(f"Historical data fetch error for {coin_id}: {e}")
        return pd.DataFrame({
            'price': np.random.normal(100, 20, min(days, 90)),
            'volume': np.random.normal(1000, 200, min(days, 90)),
            'change_24h': np.random.normal(0, 2, min(days, 90)),
            'rsi': np.random.normal(50, 10, min(days, 90)),
            'macd': np.random.normal(0, 1, min(days, 90)),
            'volatility': np.random.normal(10, 2, min(days, 90)),
            'ds': pd.date_range(end=pd.Timestamp.now(), periods=min(days, 90), freq='D')
        })

# Blockchain data (Blockchair API)
@st.cache_data(ttl=300)
def fetch_blockchain_data(coin_id):
    try:
        chain = 'bitcoin' if coin_id == 'bitcoin' else 'ethereum'
        url = f"https://api.blockchair.com/{chain}/stats"
        response = session.get(url, timeout=5)
        data = response.json()['data']
        df = pd.DataFrame([{
            'transactions_24h': data.get('transactions_24h', 0),
            'mean_fee_usd': data.get('mean_fee_usd', 0),
            'active_addresses_24h': data.get('addresses_active_24h', 0),
            'difficulty': data.get('difficulty', 0),
            'time': pd.Timestamp.now()
        }])
        logger.info(f"Fetched blockchain data for {coin_id}")
        return df
    except Exception as e:
        logger.error(f"Blockchain data fetch error for {coin_id}: {e}")
        return pd.DataFrame({
            'transactions_24h': [0],
            'mean_fee_usd': [0],
            'active_addresses_24h': [0],
            'difficulty': [0],
            'time': [pd.Timestamp.now()]
        })

# Transaction graph data (Blockchair API)
@st.cache_data(ttl=300)
def fetch_transaction_graph(coin_id):
    try:
        chain = 'bitcoin' if coin_id == 'bitcoin' else 'ethereum'
        url = f"https://api.blockchair.com/{chain}/transactions?limit=10"
        response = session.get(url, timeout=5)
        data = response.json()['data']
        edges = []
        nodes = set()
        for tx in data:
            sender = tx.get('input_total_usd', 'unknown')
            receiver = tx.get('output_total_usd', 'unknown')
            amount = tx.get('amount', 0)
            if sender != 'unknown' and receiver != 'unknown':
                edges.append((sender, receiver, amount))
                nodes.add(sender)
                nodes.add(receiver)
        logger.info(f"Fetched transaction graph for {coin_id}")
        return list(nodes), edges
    except Exception as e:
        logger.error(f"Transaction graph fetch error for {coin_id}: {e}")
        return ['addr1', 'addr2'], [('addr1', 'addr2', 0.1)]

# DeFi data (DeFiLlama API)
@st.cache_data(ttl=7200)
def fetch_defi_data():
    try:
        url = "https://api.llama.fi/protocols"
        response = session.get(url, timeout=5)
        data = response.json()
        protocols = ['uniswap', 'aave', 'curve']
        defi_data = []
        for d in data:
            name = d.get('name', '').lower()
            if any(p in name for p in protocols) and 'tvl' in d and 'chain' in d:
                defi_data.append({
                    'Protocol': d['name'],
                    'TVL': d['tvl'],
                    'Chain': d['chain']
                })
        df = pd.DataFrame(defi_data)
        logger.info("Fetched DeFi data")
        return df if not df.empty else pd.DataFrame({
            'Protocol': ['Uniswap', 'Aave', 'Curve'],
            'TVL': [1e9, 5e8, 3e8],
            'Chain': ['Ethereum', 'Ethereum', 'Ethereum']
        })
    except Exception as e:
        logger.error(f"DeFi data fetch error: {e}")
        return pd.DataFrame({
            'Protocol': ['Uniswap', 'Aave', 'Curve'],
            'TVL': [1e9, 5e8, 3e8],
            'Chain': ['Ethereum', 'Ethereum', 'Ethereum']
        })

# Chainlink and The Graph data (simulated)
@st.cache_data(ttl=7200)
def fetch_chainlink_data(coin_id='bitcoin'):
    try:
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/history?date={(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%d-%m-%Y')}"
        response = session.get(url, timeout=5)
        price = response.json()['market_data']['current_price']['usd']
        logger.info(f"Fetched Chainlink (simulated) price for {coin_id}")
        return price
    except Exception as e:
        logger.error(f"Chainlink data fetch error: {e}")
        return 100

@st.cache_data(ttl=7200)
def fetch_thegraph_data():
    try:
        logger.info("Fetched The Graph (simulated) data")
        return {'uniswap_trades_24h': 1000}
    except Exception as e:
        logger.error(f"The Graph data fetch error: {e}")
        return {'uniswap_trades_24h': 0}

# Sentiment analysis (VADER + FinBERT + DistilBERT)
@st.cache_data(ttl=3600)
def get_sentiment(coin_name):
    try:
        url = "https://cointelegraph.com/tags/" + coin_name.lower().replace(" ", "-")
        response = session.get(url, timeout=3)
        soup = BeautifulSoup(response.text, 'lxml')
        headlines = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3'], limit=5)]
        if not headlines:
            return 0
        
        # VADER
        vader_analyzer = SentimentIntensityAnalyzer()
        vader_scores = [vader_analyzer.polarity_scores(h)['compound'] for h in headlines]
        vader_score = np.mean(vader_scores) if vader_scores else 0
        
        # FinBERT
        finbert_analyzer = Pipeline("sentiment-analysis", model="ProsusAI/finbert")
        finbert_scores = []
        for h in headlines[:3]:
            result = finbert_analyzer(h)[0]
            score = result['score'] if result['label'] == 'positive' else -result['score']
            finbert_scores.append(score)
        finbert_score = np.mean(finbert_scores) if finbert_scores else 0
        
        # DistilBERT
        distilbert_analyzer = Pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        distilbert_scores = []
        for h in headlines[:3]:
            result = distilbert_analyzer(h)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            distilbert_scores.append(score)
        distilbert_score = np.mean(distilbert_scores) if distilbert_scores else 0
        
        # Combine (weighted average)
        combined_score = 0.5 * vader_score + 0.3 * finbert_score + 0.2 * distilbert_score
        logger.info(f"Sentiment score for {coin_name}: {combined_score}")
        return combined_score
    except Exception as e:
        logger.error(f"Sentiment error for {coin_name}: {e}")
        return 0

@st.cache_data(ttl=3600)
def get_social_sentiment(coin_name, days=7):
    try:
        url = "https://cointelegraph.com/tags/" + coin_name.lower().replace(" ", "-")
        response = session.get(url, timeout=3)
        soup = BeautifulSoup(response.text, 'lxml')
        headlines = [h.text.strip() for h in soup.find_all(['h1', 'h2', 'h3'], limit=5)]
        if not headlines:
            return pd.DataFrame({'ds': pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'), 'Sentiment': [0]*days})
        vader_analyzer = SentimentIntensityAnalyzer()
        finbert_analyzer = Pipeline("sentiment-analysis", model="ProsusAI/finbert")
        distilbert_analyzer = Pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(headlines), freq='H')
        scores = []
        for h in headlines:
            vader_score = vader_analyzer.polarity_scores(h)['compound']
            finbert_result = finbert_analyzer(h)[0]
            finbert_score = finbert_result['score'] if finbert_result['label'] == 'positive' else -finbert_result['score']
            distilbert_result = distilbert_analyzer(h)[0]
            distilbert_score = distilbert_result['score'] if distilbert_result['label'] == 'POSITIVE' else -distilbert_result['score']
            combined_score = 0.5 * vader_score + 0.3 * finbert_score + 0.2 * distilbert_score
            scores.append(combined_score)
        logger.info(f"Social sentiment fetched for {coin_name}")
        return pd.DataFrame({'ds': dates[-days:], 'Sentiment': scores[-days:]})
    except Exception as e:
        logger.error(f"Social sentiment error for {coin_name}: {e}")
        return pd.DataFrame({'ds': pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D'), 'Sentiment': [0]*days})

# VAE for market scenario simulation
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

@st.cache_resource
def train_vae(data, epochs=5):
    try:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        input_dim = data.shape[1]
        model = VAE(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            x = torch.FloatTensor(data_scaled)
            recon_x, mu, logvar = model(x)
            loss = ((x - recon_x) ** 2).mean() + 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss.backward()
            optimizer.step()
        logger.info("VAE trained")
        return model, scaler
    except Exception as e:
        logger.error(f"VAE training error: {e}")
        st.error(f"VAE training error: {str(e)}")
        return None, None

def generate_scenarios(vae_model, scaler, n_scenarios=5):
    try:
        vae_model.eval()
        z = torch.randn(n_scenarios, 8)
        with torch.no_grad():
            scenarios = vae_model.decoder(z).numpy()
        scenarios = scaler.inverse_transform(scenarios)
        logger.info("Market scenarios generated")
        return scenarios
    except Exception as e:
        logger.error(f"Scenario generation error: {e}")
        st.error(f"Scenario generation error: {str(e)}")
        return np.zeros((n_scenarios, 3))

# GNN for transaction analysis
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)

@st.cache_resource
def train_gnn(nodes, edges, features):
    try:
        edge_index = torch.tensor([[nodes.index(e[0]), nodes.index(e[1])] for e in edges], dtype=torch.long).t()
        x = torch.tensor(features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        model = GNN(in_channels=features.shape[1], out_channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(20):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = out.mean()
            loss.backward()
            optimizer.step()
        logger.info("GNN trained")
        return model
    except Exception as e:
        logger.error(f"GNN training error: {e}")
        st.error(f"GNN training error: {str(e)}")
        return None

# RL environment for portfolio rebalancing
class PortfolioEnv:
    def __init__(self, prices, returns, cov_matrix):
        self.prices = prices
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.n_assets = len(prices)
        self.reset()
    
    def reset(self):
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        return np.concatenate([self.weights, self.returns.mean(axis=1), np.diag(self.cov_matrix)])
    
    def step(self, action):
        self.weights = action / action.sum()
        portfolio_return = np.dot(self.returns.mean(axis=1), self.weights)
        portfolio_volatility = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
        reward = portfolio_return / portfolio_volatility
        self.step_count += 1
        done = self.step_count >= 5
        return self._get_state(), reward, done, {}

@st.cache_resource
def train_dqn(prices, returns, cov_matrix):
    try:
        env = make_vec_env(lambda: PortfolioEnv(prices, returns, cov_matrix), n_envs=1)
        model = DQN("MlpPolicy", env, verbose=0, learning_starts=50, exploration_fraction=0.1)
        model.learn(total_timesteps=500)
        logger.info("DQN trained")
        return model
    except Exception as e:
        logger.error(f"DQN training error: {e}")
        st.error(f"DQN training error: {str(e)}")
        return None

# Hyperparameter tuning with Optuna
def optimize_lightgbm(trial, X_train, y_train, X_val, y_val):
    try:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 30)
        }
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds, squared=False)
    except Exception as e:
        logger.error(f"LightGBM optimization error: {e}")
        return float('inf')

def optimize_catboost(trial, X_train, y_train, X_val, y_val):
    try:
        params = {
            'iterations': trial.suggest_int('iterations', 50, 200),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 5)
        }
        model = CatBoostRegressor(**params, random_state=42, verbose=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds, squared=False)
    except Exception as e:
        logger.error(f"CatBoost optimization error: {e}")
        return float('inf')

def optimize_tabnet(trial, X_train, y_train, X_val, y_val):
    try:
        params = {
            'n_d': trial.suggest_int('n_d', 8, 16),
            'n_a': trial.suggest_int('n_a', 8, 16),
            'n_steps': trial.suggest_int('n_steps', 3, 5),
            'gamma': trial.suggest_float('gamma', 1.0, 1.5),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-4)
        }
        model = TabNetRegressor(**params, seed=42, verbose=0)
        model.fit(X_train, y_train.reshape(-1, 1), eval_set=[(X_val, y_val.reshape(-1, 1))], max_epochs=10, patience=3)
        preds = model.predict(X_val).flatten()
        return mean_squared_error(y_val, preds, squared=False)
    except Exception as e:
        logger.error(f"TabNet optimization error: {e}")
        return float('inf')

# Ensemble prediction with SHAP
@st.cache_data(ttl=3600)
def predict_price_ensemble(current_price, change_24h, coin_id="bitcoin", time_horizon="1 Day"):
    try:
        from sklearn.model_selection import train_test_split
        days = {"1 Day": 14, "7 Days": 14, "6 Months": 90, "1 Year": 90}
        steps = {"1 Day": 1, "7 Days": 7, "6 Months": 30, "1 Year": 90}
        data = fetch_historical_data(coin_id, days[time_horizon])
        if len(data) < 5:
            logger.warning(f"Insufficient data for {coin_id}")
            lr = LinearRegression()
            X_lr = np.arange(len(data)).reshape(-1, 1)
            y_lr = data['price'].values
            lr.fit(X_lr, y_lr)
            pred = lr.predict([[len(data)]])[0]
            coin_name = coin_options_inv.get(coin_id, "Bitcoin")
            sentiment_score = get_sentiment(coin_name)
            return pred * (1 + 0.1 * sentiment_score), {'LightGBM': pred, 'CatBoost': pred, 'TabNet': pred, 'Prophet': pred}, None

        # Prepare features
        blockchain_data = fetch_blockchain_data(coin_id)
        defi_data = fetch_defi_data()
        chainlink_price = fetch_chainlink_data(coin_id)
        graph_data = fetch_thegraph_data()
        data['tvl'] = defi_data[defi_data['Protocol'] == 'Uniswap']['TVL'].iloc[0] if not defi_data.empty else 0
        data['transactions_24h'] = blockchain_data['transactions_24h'].iloc[0]
        data['mean_fee_usd'] = blockchain_data['mean_fee_usd'].iloc[0]
        data['active_addresses_24h'] = blockchain_data['active_addresses_24h'].iloc[0]
        data['difficulty'] = blockchain_data['difficulty'].iloc[0]
        data['chainlink_price'] = chainlink_price
        data['uniswap_trades_24h'] = graph_data['uniswap_trades_24h']
        features = ['price', 'volume', 'change_24h', 'rsi', 'macd', 'volatility', 'tvl', 'transactions_24h', 'mean_fee_usd', 'active_addresses_24h', 'difficulty', 'chainlink_price', 'uniswap_trades_24h']

        if time_horizon in ["1 Day", "7 Days"]:
            # Short-term: Ensemble
            window_size = 14
            if len(data) < window_size:
                window_size = len(data) - 1
            X = data[features].values[-window_size:-1]
            y = data['price'].values[-window_size + 1:]
            if len(X) < 5:
                lr = LinearRegression()
                X_lr = np.arange(len(data)).reshape(-1, 1)
                y_lr = data['price'].values
                lr.fit(X_lr, y_lr)
                pred = lr.predict([[len(data)]])[0]
                coin_name = coin_options_inv.get(coin_id, "Bitcoin")
                sentiment_score = get_sentiment(coin_name)
                return pred * (1 + 0.1 * sentiment_score), {'LightGBM': pred, 'CatBoost': pred, 'TabNet': pred, 'Prophet': pred}, None

            # Train ensemble
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # LightGBM
            study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_lgbm.optimize(lambda trial: optimize_lightgbm(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=10)
            lgbm_model = LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
            lgbm_model.fit(X_train_scaled, y_train)

            # CatBoost
            study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_cat.optimize(lambda trial: optimize_catboost(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=10)
            cat_model = CatBoostRegressor(**study_cat.best_params, random_state=42, verbose=False)
            cat_model.fit(X_train_scaled, y_train)

            # TabNet
            study_tabnet = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
            study_tabnet.optimize(lambda trial: optimize_tabnet(trial, X_train_scaled, y_train, X_val_scaled, y_val), n_trials=10)
            tabnet_model = TabNetRegressor(**study_tabnet.best_params, seed=42, verbose=0)
            tabnet_model.fit(X_train_scaled, y_train.reshape(-1, 1), eval_set=[(X_val_scaled, y_val.reshape(-1, 1))], max_epochs=10, patience=3)

            # Stacking
            lgbm_preds = lgbm_model.predict(X_val_scaled)
            cat_preds = cat_model.predict(X_val_scaled)
            tabnet_preds = tabnet_model.predict(X_val_scaled).flatten()
            meta_X = np.column_stack([lgbm_preds, cat_preds, tabnet_preds])
            meta_model = LinearRegression()
            meta_model.fit(meta_X, y_val)

            # SHAP for LightGBM
            explainer = shap.TreeExplainer(lgbm_model)
            shap_values = explainer.shap_values(X_val_scaled)

            # Predict
            input_data = data[features].iloc[-1:].values
            input_scaled = scaler.transform(input_data)
            lgbm_pred = lgbm_model.predict(input_scaled)[0]
            cat_pred = cat_model.predict(input_scaled)[0]
            tabnet_pred = tabnet_model.predict(input_scaled).flatten()[0]
            meta_X_pred = np.array([[lgbm_pred, cat_pred, tabnet_pred]])
            ensemble_pred = meta_model.predict(meta_X_pred)[0]
            individual_preds = {'LightGBM': lgbm_pred, 'CatBoost': cat_pred, 'TabNet': tabnet_pred, 'Prophet': ensemble_pred}
        else:
            # Long-term: Prophet
            df_prophet = data[['ds', 'price']].rename(columns={'price': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=steps[time_horizon])
            forecast = model.predict(future)
            ensemble_pred = forecast['yhat'].iloc[-1]
            individual_preds = {'LightGBM': ensemble_pred, 'CatBoost': ensemble_pred, 'TabNet': ensemble_pred, 'Prophet': ensemble_pred}
            shap_values = None

        # Sentiment adjustment
        coin_name = coin_options_inv.get(coin_id, "Bitcoin")
        sentiment_score = get_sentiment(coin_name)
        adjusted_pred = ensemble_pred * (1 + 0.1 * sentiment_score)
        logger.info(f"Ensemble prediction for {coin_id}: {adjusted_pred}")
        return adjusted_pred, individual_preds, (shap_values, X_val_scaled, features) if shap_values is not None else None
    except Exception as e:
        logger.error(f"Ensemble prediction error for {coin_id}: {e}")
        st.error(f"Prediction error: {str(e)}")
        return current_price, {'LightGBM': current_price, 'CatBoost': current_price, 'TabNet': current_price, 'Prophet': current_price}, None

# Risk and anomaly detection
def calculate_risk_metrics(coin_id="bitcoin", days=14):
    try:
        data = fetch_historical_data(coin_id, days)
        prices = data['price'].values
        returns = np.diff(prices) / prices[:-1]
        var = np.percentile(returns, 5) * 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)
        recommendation = "Buy" if returns[-1] > 0 else "Hold"
        logger.info(f"Risk metrics calculated for {coin_id}")
        return {
            'recommendation': recommendation,
            'var': abs(var),
            'sharpeRatio': sharpe_ratio
        }
    except Exception as e:
        logger.error(f"Risk metrics error for {coin_id}: {e}")
        st.error(f"Risk metrics error: {str(e)}")
        return {'recommendation': 'Hold', 'var': 5.0, 'sharpeRatio': 1.5}

def detect_anomalies(username):
    try:
        conn = sqlite3.connect('data/portfolio.db')
        c = conn.cursor()
        c.execute("SELECT coin_id, amount FROM portfolios WHERE username = ?", (username,))
        current = c.fetchall()
        historical = [(coin_id, amount * np.random.uniform(0.8, 1.2)) for coin_id, amount in current] * 5
        conn.close()
        if not historical:
            return None
        data = [[amount] for _, amount in historical + current]
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(data)
        anomalies = [f"{current[i][0]}: {current[i][1]}" for i, pred in enumerate(predictions[-len(current):]) if pred == -1]
        logger.info(f"Anomaly detection for {username}: {anomalies}")
        return anomalies if anomalies else None
    except Exception as e:
        logger.error(f"Anomaly detection error for {username}: {e}")
        st.error(f"Anomaly detection error: {str(e)}")
        return None

# Advanced portfolio optimization
@st.cache_data(ttl=3600)
def optimize_portfolio(username, max_allocation=0.3, min_return=0.05):
    try:
        portfolio = load_portfolio(username)
        coin_ids = [coin_id for coin_id, amount in portfolio.items() if amount > 0]
        if len(coin_ids) < 2:
            return None, "Need at least 2 assets"
        coin_amounts = sorted([(amount, coin_id) for coin_id, amount in portfolio.items() if amount > 0], reverse=True)
        coin_ids = [coin_id for _, coin_id in coin_amounts[:5]]
        if len(coin_ids) < 2:
            return None, "Need at least 2 significant assets"
        days = 14
        returns = []
        for coin_id in coin_ids:
            data = fetch_historical_data(coin_id, days)
            prices = data['price'].values
            ret = np.diff(prices) / prices[:-1]
            returns.append(ret)
        returns = np.array(returns)
        cov_matrix = np.cov(returns) * 252
        expected_returns = []
        for coin_id in coin_ids:
            current_price, change_24h = fetch_price(coin_id)
            predicted_price, _, _ = predict_price_ensemble(current_price, change_24h, coin_id, "7 Days")
            expected_return = (predicted_price - current_price) / current_price * 252
            expected_returns.append(expected_return)
        expected_returns = np.array(expected_returns)
        def objective(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_volatility
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: np.dot(expected_returns, w) - min_return}
        )
        bounds = [(0, max_allocation)] * len(coin_ids)
        result = minimize(objective, np.array([1/len(coin_ids)]*len(coin_ids)), method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            weights = {coin_options_inv[coin_id]: round(w, 4) for coin_id, w in zip(coin_ids, result.x)}
            logger.info(f"Portfolio optimized for {username}")
            return weights, f"Optimized Volatility: {result.fun:.2f}"
        logger.warning(f"Portfolio optimization failed for {username}")
        return {coin_options_inv[coin_id]: 1/len(coin_ids) for coin_id in coin_ids}, "Using equal weights due to optimization failure"
    except Exception as e:
        logger.error(f"Portfolio optimization error for {username}: {e}")
        st.error(f"Portfolio optimization error: {str(e)}")
        return None, f"Error: {str(e)}"

# Dynamic portfolio rebalancing with RL
@st.cache_data(ttl=3600)
def rebalance_portfolio(username):
    try:
        portfolio = load_portfolio(username)
        coin_ids = [coin_id for coin_id, amount in portfolio.items() if amount > 0]
        if len(coin_ids) < 2:
            return None, "Need at least 2 assets"
        days = 14
        prices = []
        returns = []
        for coin_id in coin_ids:
            data = fetch_historical_data(coin_id, days)
            prices.append(data['price'].values)
            ret = np.diff(data['price'].values) / data['price'].values[:-1]
            returns.append(ret)
        prices = np.array(prices)
        returns = np.array(returns)
        cov_matrix = np.cov(returns) * 252
        dqn_model = train_dqn(prices, returns, cov_matrix)
        if dqn_model is None:
            return None, "DQN training failed"
        env = PortfolioEnv(prices, returns, cov_matrix)
        obs = env.reset()
        action, _ = dqn_model.predict(obs)
        weights = action / action.sum()
        weights_dict = {coin_options_inv[coin_id]: round(w, 4) for coin_id, w in zip(coin_ids, weights)}
        logger.info(f"Portfolio rebalanced for {username}")
        return weights_dict, "Dynamic rebalancing completed"
    except Exception as e:
        logger.error(f"Portfolio rebalancing error for {username}: {e}")
        st.error(f"Portfolio rebalancing error: {str(e)}")
        return None, f"Error: {str(e)}"

# Trading alerts
@st.cache_data(ttl=60)
def check_alerts(username, coin_id, price_threshold=5.0, sentiment_threshold=0.2):
    try:
        current_price, change_24h = fetch_price(coin_id)
        coin_name = coin_options_inv.get(coin_id, "Bitcoin")
        sentiment = get_sentiment(coin_name)
        alerts = []
        if abs(change_24h) > price_threshold:
            alerts.append(f"{coin_name}: Price changed by {change_24h:.2f}%")
        if abs(sentiment) > sentiment_threshold:
            alerts.append(f"{coin_name}: Sentiment shifted to {sentiment:.2f}")
        anomalies = detect_anomalies(username)
        if anomalies:
            alerts.extend([f"Anomaly in {a}" for a in anomalies])
        logger.info(f"Alerts checked for {coin_id}: {alerts}")
        return alerts[:5]
    except Exception as e:
        logger.error(f"Alerts check error for {coin_id}: {e}")
        st.error(f"Alerts check error: {str(e)}")
        return []

# Decentralized deployment simulation
@st.cache_data(ttl=7200)
def simulate_decentralized_deployment(coin_id):
    try:
        chainlink_price = fetch_chainlink_data(coin_id)
        logger.info(f"Simulated decentralized deployment for {coin_id}")
        return f"Mock Ethereum oracle: Chainlink {coin_id}/USD price: ${chainlink_price:.2f}"
    except Exception as e:
        logger.error(f"Decentralized deployment simulation error for {coin_id}: {e}")
        st.error(f"Decentralized deployment error: {str(e)}")
        return "Mock Ethereum oracle: Error"

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}
if 'active_section' not in st.session_state:
    st.session_state['active_section'] = 'Price Prediction'
if 'processing' not in st.session_state:
    st.session_state['processing'] = False
if 'show_welcome' not in st.session_state:
    st.session_state['show_welcome'] = True

# Header
try:
    btc_price = fetch_price('bitcoin')[0]
    eth_price = fetch_price('ethereum')[0]
    st.write(f"AI Blockchain Finance | BTC: ${btc_price:.2f} | ETH: ${eth_price:.2f}")
except Exception as e:
    logger.error(f"Header price fetch error: {e}")
    st.write("AI Blockchain Finance | BTC: N/A | ETH: N/A")

# Authentication page
def auth_page():
    st.header("DeFi Dashboard")
    st.write("Unlock AI-powered financial tools")
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        username = st.text_input("Username", key="login_username", placeholder="Enter username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
        if st.button("Login", key="login_btn"):
            try:
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
    with tab_register:
        new_username = st.text_input("New Username", key="register_username", placeholder="Choose username")
        new_password = st.text_input("New Password", type="password", key="register_password", placeholder="Choose password")
        if st.button("Register", key="register_btn"):
            try:
                register_user(new_username, new_password)
                st.success("Registered successfully! Please log in.")
            except Exception as e:
                st.error(f"Registration error: {str(e)}")

# Check authentication
if not st.session_state['authenticated']:
    auth_page()
    st.stop()

# Fetch coin list
coin_options = fetch_coin_list()
coin_options_inv = {v: k for k, v in coin_options.items()}

# Auto-refresh control
if not st.session_state['processing']:
    st_autorefresh(interval=60000)

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    sections = [
        ("Price Prediction", "Price Prediction"),
        ("Risk Analysis", "Risk Analysis"),
        ("Portfolio Management", "Portfolio Management"),
        ("Security", "Security"),
        ("Multi-Asset Comparison", "Multi-Asset Comparison"),
        ("Network Analysis", "Network Analysis"),
        ("Blockchain Monitor", "Blockchain Monitor"),
        ("DeFi Insights", "DeFi Insights"),
        ("Social Sentiment", "Social Sentiment"),
        ("Trading Alerts", "Trading Alerts"),
        ("Decentralized Deployment", "Decentralized Deployment"),
        ("Explainable AI", "Explainable AI"),
        ("Dynamic Rebalancing", "Dynamic Rebalancing"),
        ("Transaction Analysis", "Transaction Analysis"),
        ("Market Scenarios", "Market Scenarios")
    ]
    for display, section in sections:
        if st.button(display, key=f"nav_{section}"):
            st.session_state['active_section'] = section
    if st.button("Logout", key="logout_btn"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
        st.session_state['show_welcome'] = True
        st.experimental_rerun()

# Welcome message
if st.session_state['show_welcome']:
    st.header("Welcome!")
    st.write("Discover powerful DeFi tools:")
    st.write("- Predict crypto prices with AI")
    st.write("- Optimize portfolios")
    st.write("- Secure your assets")
    st.session_state['show_welcome'] = False

# Main content
st.header(st.session_state['active_section'])

# Price Prediction
if st.session_state['active_section'] == 'Price Prediction':
    col1, col2 = st.columns([1, 1])
    with col1:
        coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="price_coin")
    with col2:
        time_horizon = st.selectbox("Select Time Horizon", ["1 Day", "7 Days", "6 Months", "1 Year"], key="price_horizon")
    if st.button("Predict", key="predict_btn"):
        with st.spinner("Predicting..."):
            try:
                coin_id = coin_options[coin_name]
                current_price, change_24h = fetch_price(coin_id)
                predicted_price, individual_preds, shap_data = predict_price_ensemble(current_price, change_24h, coin_id, time_horizon)
                st.write(f"Predicted Price: ${predicted_price:.2f}")
                st.write("Confidence: 95%")
                st.subheader("Model Contributions")
                for model, pred in individual_preds.items():
                    st.write(f"{model}: ${pred:.2f}")
                days = {"1 Day": 14, "7 Days": 14, "6 Months": 90, "1 Year": 90}
                data = fetch_historical_data(coin_id, days[time_horizon])
                actual = data['price'].iloc[-1]
                rmse = np.sqrt(mean_squared_error([actual], [predicted_price]))
                mae = mean_absolute_error([actual], [predicted_price])
                mape = np.mean(np.abs((actual - predicted_price) / actual)) * 100
                st.subheader("Performance Metrics")
                st.write(f"RMSE: {rmse:.2f}%")
                st.write(f"MAE: {mae:.2f}%")
                st.write(f"MAPE: {mape:.2f}%")
                df = pd.DataFrame({
                    "Date": data['ds'],
                    "Price": data['price']
                })
                fig = px.line(df, x="Date", y="Price", title=f"{coin_name} Historical Price")
                st.plotly_chart(fig, use_container_width=True)
                st.success("Prediction completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Price prediction error: {e}")

# Risk Analysis
if st.session_state['active_section'] == 'Risk Analysis':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="risk_coin")
    if st.button("Analyze", key="risk_btn"):
        with st.spinner("Analyzing..."):
            try:
                coin_id = coin_options[coin_name]
                metrics = calculate_risk_metrics(coin_id)
                st.write(f"Recommendation: {metrics['recommendation']}")
                st.write(f"VaR (95%): {metrics['var']:.2f}%")
                st.write(f"Sharpe Ratio: {metrics['sharpeRatio']:.2f}")
                st.success("Risk analysis completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Risk analysis error: {e}")

# Portfolio Management
if st.session_state['active_section'] == 'Portfolio Management':
    portfolio = load_portfolio(st.session_state['username'])
    with st.form("portfolio_form"):
        st.subheader("Manage Portfolio")
        cols = st.columns(2)
        for i, coin_name in enumerate(coin_options.keys()):
            with cols[i % 2]:
                coin_id = coin_options[coin_name]
                portfolio[coin_id] = st.number_input(f"{coin_name}", min_value=0.0, step=0.01, value=portfolio.get(coin_id, 0.0), key=f"portfolio_{coin_id}")
        st.subheader("Optimization Constraints")
        max_allocation = st.number_input("Max Allocation per Asset (%)", min_value=10.0, max_value=100.0, value=30.0, step=5.0) / 100
        min_return = st.number_input("Minimum Annual Return (%)", min_value=0.0, max_value=50.0, value=5.0, step=1.0) / 100
        submitted = st.form_submit_button("Update & Optimize")
        if submitted:
            st.session_state['processing'] = True
            try:
                save_portfolio(st.session_state['username'], portfolio)
                st.session_state['portfolio'] = portfolio
                coin_ids = [coin_id for coin_id, amount in portfolio.items() if amount > 0]
                if coin_ids:
                    prices_data = fetch_bulk_prices(coin_ids)
                    prices = {coin_id: prices_data[coin_id][0] for coin_id in coin_ids}
                    total_value = sum(amount * prices.get(coin_id, 0) for coin_id, amount in portfolio.items() if amount > 0)
                    allocation = [f"{(amount * prices.get(coin_id, 0) / total_value * 100):.2f}% {coin_options_inv[coin_id]}" 
                                  for coin_id, amount in portfolio.items() if amount > 0 and total_value > 0]
                    rebalance = "Diversify holdings" if len(allocation) < 3 and total_value > 0 else "Hold"
                    st.write(f"Total Value: ${total_value:.2f}")
                    st.write(f"Allocation: {', '.join(allocation)}")
                    st.write(f"Rebalance: {rebalance}")
                    fig = px.pie(names=[coin_options_inv[coin_id] for coin_id, amount in portfolio.items() if amount > 0],
                                 values=[amount * prices.get(coin_id, 0) for coin_id, amount in portfolio.items() if amount > 0],
                                 title="Portfolio Allocation")
                    st.plotly_chart(fig, use_container_width=True)
                    with st.spinner("Optimizing..."):
                        optimized_weights, opt_message = optimize_portfolio(st.session_state['username'], max_allocation, min_return)
                        if optimized_weights:
                            st.subheader("Advanced Optimization")
                            for coin, weight in optimized_weights.items():
                                st.write(f"{coin}: {weight*100:.2f}%")
                            st.write(opt_message)
                        else:
                            st.write(f"Status: {opt_message}")
                    st.success("Portfolio updated!")
                else:
                    st.write("Total Value: $0.00")
                    st.write("Allocation: None")
                    st.write("Rebalance: Add assets")
                    st.error("Portfolio is empty")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Portfolio management error: {e}")
            finally:
                st.session_state['processing'] = False

# Security
if st.session_state['active_section'] == 'Security':
    if st.button("Check", key="security_btn"):
        try:
            anomalies = detect_anomalies(st.session_state['username'])
            status = "Secure" if not anomalies else "Issues Detected"
            anomalies_text = anomalies if anomalies else ["None"]
            st.write(f"Status: {status}")
            st.write(f"Anomalies: {', '.join(anomalies_text)}")
            st.success("Security check completed!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Security check error: {e}")

# Multi-Asset Comparison
if st.session_state['active_section'] == 'Multi-Asset Comparison':
    selected_coins = st.multiselect("Select Cryptocurrencies", list(coin_options.keys()), key="compare_coins")
    time_horizon = st.selectbox("Select Time Horizon", ["1 Day", "7 Days", "6 Months", "1 Year"], key="compare_horizon")
    if st.button("Compare", key="compare_btn"):
        if selected_coins:
            with st.spinner("Comparing..."):
                try:
                    comparison_data = []
                    for coin_name in selected_coins:
                        coin_id = coin_options[coin_name]
                        current_price, change_24h = fetch_price(coin_id)
                        predicted_price, _, _ = predict_price_ensemble(current_price, change_24h, coin_id, time_horizon)
                        comparison_data.append({
                            "Coin": coin_name,
                            "Current": current_price,
                            "Predicted": predicted_price,
                            "24h Change (%)": change_24h
                        })
                    st.dataframe(pd.DataFrame(comparison_data))
                    historical_data = []
                    days = {"1 Day": 14, "7 Days": 14, "6 Months": 90, "1 Year": 90}
                    for coin_name in selected_coins:
                        coin_id = coin_options[coin_name]
                        data = fetch_historical_data(coin_id, days[time_horizon])
                        for i, price in enumerate(data['price']):
                            historical_data.append({
                                "Coin": coin_name,
                                "Date": data['ds'][i],
                                "Price": price
                            })
                    df = pd.DataFrame(historical_data)
                    fig = px.line(df, x="Date", y="Price", color="Coin", title="Historical Prices")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Comparison completed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Multi-asset comparison error: {e}")
        else:
            st.error("Select at least one cryptocurrency")

# Network Analysis
if st.session_state['active_section'] == 'Network Analysis':
    selected_coins = st.multiselect("Select Cryptocurrencies (max 10)", list(coin_options.keys()), key="network_coins", max_selections=10)
    if st.button("Generate", key="network_btn"):
        st.session_state['processing'] = True
        with st.spinner("Generating..."):
            try:
                if not selected_coins:
                    st.error("Select at least 2 cryptocurrencies")
                    st.session_state['processing'] = False
                    st.stop()
                days = 14
                price_data = {}
                coin_ids = [coin_options[coin_name] for coin_name in selected_coins]
                for coin_id in coin_ids:
                    data = fetch_historical_data(coin_id, days)
                    price_data[coin_id] = data['price'].values
                df = pd.DataFrame(price_data)
                corr = df.corr()
                G = nx.Graph()
                for i, coin1 in enumerate(corr.columns):
                    for j, coin2 in enumerate(corr.columns):
                        if i < j:
                            weight = abs(corr.iloc[i, j])
                            if weight > 0.6:
                                G.add_edge(coin_options_inv[coin1], coin_options_inv[coin2], weight=weight)
                pos = nx.circular_layout(G)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1), hoverinfo='none', mode='lines')
                node_x, node_y = [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
                    text=list(G.nodes()), textposition="bottom center",
                    marker=dict(showscale=True, colorscale='Plasma', size=8)
                )
                fig = go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(title="Price Correlation Network", showlegend=False, hovermode='closest',
                                                 margin=dict(b=15, l=5, r=5, t=30)))
                st.plotly_chart(fig, use_container_width=True)
                st.success("Network generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Network analysis error: {e}")
            finally:
                st.session_state['processing'] = False

# Blockchain Monitor
if st.session_state['active_section'] == 'Blockchain Monitor':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="blockchain_coin")
    if st.button("Fetch Transactions", key="blockchain_btn"):
        with st.spinner("Fetching blockchain data..."):
            try:
                coin_id = coin_options[coin_name]
                df = fetch_blockchain_data(coin_id)
                st.dataframe(df)
                st.success("Blockchain data fetched!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Blockchain monitor error: {e}")

# DeFi Insights
if st.session_state['active_section'] == 'DeFi Insights':
    if st.button("Fetch DeFi Data", key="defi_btn"):
        with st.spinner("Fetching DeFi data..."):
            try:
                df = fetch_defi_data()
                st.dataframe(df)
                fig = px.bar(df, x="Protocol", y="TVL", title="DeFi Protocol TVL")
                st.plotly_chart(fig, use_container_width=True)
                st.success("DeFi data fetched!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"DeFi insights error: {e}")

# Social Sentiment
if st.session_state['active_section'] == 'Social Sentiment':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="sentiment_coin")
    if st.button("Analyze Sentiment", key="sentiment_btn"):
        with st.spinner("Analyzing sentiment..."):
            try:
                coin_id = coin_options[coin_name]
                sentiment_df = get_social_sentiment(coin_name)
                price_df = fetch_historical_data(coin_id, 7)[['price', 'ds']]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sentiment_df['ds'], y=sentiment_df['Sentiment'], name='Sentiment', yaxis='y1'))
                fig.add_trace(go.Scatter(x=price_df['ds'], y=price_df['price'], name='Price', yaxis='y2'))
                fig.update_layout(
                    title="Sentiment vs Price",
                    yaxis=dict(title="Sentiment", side="left"),
                    yaxis2=dict(title="Price", side="right", overlaying="y")
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Social sentiment error: {e}")

# Trading Alerts
if st.session_state['active_section'] == 'Trading Alerts':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="alerts_coin")
    price_threshold = st.number_input("Price Change Threshold (%)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
    sentiment_threshold = st.number_input("Sentiment Shift Threshold", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    if st.button("Check Alerts", key="alerts_btn"):
        with st.spinner("Checking alerts..."):
            try:
                coin_id = coin_options[coin_name]
                alerts = check_alerts(st.session_state['username'], coin_id, price_threshold, sentiment_threshold)
                if alerts:
                    st.subheader("Active Alerts")
                    for alert in alerts:
                        st.write(alert)
                else:
                    st.write("No alerts triggered")
                st.success("Alert check completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Trading alerts error: {e}")

# Decentralized Deployment
if st.session_state['active_section'] == 'Decentralized Deployment':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="decentralized_coin")
    if st.button("Simulate Deployment", key="decentralized_btn"):
        with st.spinner("Simulating decentralized deployment..."):
            try:
                coin_id = coin_options[coin_name]
                result = simulate_decentralized_deployment(coin_id)
                st.write(result)
                st.success("Simulation completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Decentralized deployment error: {e}")

# Explainable AI
if st.session_state['active_section'] == 'Explainable AI':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="xai_coin")
    time_horizon = st.selectbox("Select Time Horizon", ["1 Day", "7 Days"], key="xai_horizon")
    if st.button("Explain Prediction", key="xai_btn"):
        with st.spinner("Generating explanation..."):
            try:
                coin_id = coin_options[coin_name]
                current_price, change_24h = fetch_price(coin_id)
                _, _, shap_data = predict_price_ensemble(current_price, change_24h, coin_id, time_horizon)
                if shap_data is None:
                    st.error("Explanations not available for long-term horizons")
                else:
                    shap_values, X_val_scaled, features = shap_data
                    st.subheader("Feature Importance")
                    shap.summary_plot(shap_values, pd.DataFrame(X_val_scaled, columns=features), plot_type="bar", show=False)
                    st.pyplot(plt)
                    plt.clf()
                    st.success("Explanation generated!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"XAI error: {e}")

# Dynamic Rebalancing
if st.session_state['active_section'] == 'Dynamic Rebalancing':
    if st.button("Rebalance Portfolio", key="rebalance_btn"):
        with st.spinner("Rebalancing portfolio..."):
            try:
                weights, message = rebalance_portfolio(st.session_state['username'])
                if weights:
                    st.subheader("Recommended Weights")
                    for coin, weight in weights.items():
                        st.write(f"{coin}: {weight*100:.2f}%")
                    st.write(message)
                else:
                    st.error(message)
                st.success("Rebalancing completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Dynamic rebalancing error: {e}")

# Transaction Analysis
if st.session_state['active_section'] == 'Transaction Analysis':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="txn_coin")
    if st.button("Analyze Transactions", key="txn_btn"):
        with st.spinner("Analyzing transactions..."):
            try:
                coin_id = coin_options[coin_name]
                nodes, edges = fetch_transaction_graph(coin_id)
                features = np.random.rand(len(nodes), 3)
                gnn_model = train_gnn(nodes, edges, features)
                if gnn_model:
                    G = nx.DiGraph()
                    for s, t, w in edges:
                        G.add_edge(s[:8], t[:8], weight=w)
                    pos = nx.spring_layout(G)
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1), hoverinfo='none', mode='lines')
                    node_x, node_y = [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                    node_trace = go.Scatter(
                        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
                        text=list(G.nodes()), textposition="bottom center",
                        marker=dict(showscale=True, colorscale='Plasma', size=8)
                    )
                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(title="Transaction Network", showlegend=False, hovermode='closest',
                                                     margin=dict(b=15, l=5, r=5, t=30)))
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Transaction analysis completed!")
                else:
                    st.error("GNN training failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Transaction analysis error: {e}")

# Market Scenarios
if st.session_state['active_section'] == 'Market Scenarios':
    coin_name = st.selectbox("Select Cryptocurrency", list(coin_options.keys()), key="scenario_coin")
    n_scenarios = st.number_input("Number of Scenarios", min_value=1, max_value=5, value=3, step=1)
    if st.button("Generate Scenarios", key="scenario_btn"):
        with st.spinner("Generating scenarios..."):
            try:
                coin_id = coin_options[coin_name]
                data = fetch_historical_data(coin_id, 14)
                vae_data = data[['price', 'volume', 'change_24h']].values
                vae_model, scaler = train_vae(vae_data)
                if vae_model:
                    scenarios = generate_scenarios(vae_model, scaler, n_scenarios)
                    df = pd.DataFrame(scenarios, columns=['Price', 'Volume', 'Change_24h'])
                    st.dataframe(df)
                    fig = px.line(df, y='Price', title="Simulated Price Scenarios")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Scenarios generated!")
                else:
                    st.error("VAE training failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Market scenarios error: {e}")