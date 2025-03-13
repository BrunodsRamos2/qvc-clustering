import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# Gerar dados de exemplo
n_samples = 50
n_features = 2
n_clusters = 2
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normalizar

# Inicializar centróides com K-means++
def kmeans_plus_plus(X, K):
    centroids = [X[np.random.randint(X.shape[0])]]
    for _ in range(K-1):
        dists = pairwise_distances(X, centroids).min(axis=1)
        probs = dists / dists.sum()
        centroids.append(X[np.argmax(probs)])
    return np.array(centroids)

centroids = kmeans_plus_plus(X, n_clusters)

# Parâmetros quânticos
n_qubits = n_features  # Número de qubits = número de features
dev = qml.device("default.qubit", wires=2*n_qubits + 1)  # 2n_qubits para dados + 1 ancilla

# Mapa de características quântico variacional
def feature_map(x, theta, wires):
    print(wires, theta)
    for i in wires:
        qml.RX(x[i % len(x)], wires=i)  # Codificação dos dados
        qml.RY(theta[i % len(x)], wires=i)       # Parâmetros variacionais
    # Camada de entrelaçamento
    for i in range(len(wires)-1):
        qml.CZ(wires=[wires[i], wires[i+1]])

# Circuito para estimar o kernel (fidelidade via Swap Test)
@qml.qnode(dev)
def quantum_kernel(x, c, theta):
    # Codificar x nos primeiros n_qubits
    feature_map(x, theta, wires=range(n_qubits))
    # Codificar c nos próximos n_qubits
    feature_map(c, theta, wires=range(n_qubits, 2*n_qubits))
    # Swap Test para estimar |<x|c>|^2
    qml.Hadamard(wires=2*n_qubits)
    for i in range(n_qubits):
        qml.CSWAP(wires=[2*n_qubits, i, n_qubits + i])
    qml.Hadamard(wires=2*n_qubits)
    return qml.expval(qml.PauliZ(2*n_qubits))  # Medir fidelidade

# Função de custo (sobreposição entre clusters)
def cost(theta, centroids):
    custo = 0.0
    # Calcular sobreposição entre todos os pares de centróides
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            custo += quantum_kernel(centroids[i], centroids[j], theta)
    return custo

# Otimização clássica dos parâmetros theta
opt = qml.RMSPropOptimizer(stepsize=0.1)
theta = np.random.normal(0, np.pi, size=(n_qubits,), requires_grad=True)

# Loop de treinamento
max_iterations = 50
for it in range(max_iterations):
    # Atribuir clusters (etapa clássica)
    labels = []
    for x in X:
        kernels = [quantum_kernel(x, c, theta) for c in centroids]
        labels.append(np.argmax(kernels))
    
    # Atualizar centróides
    new_centroids = np.array([X[np.array(labels) == k].mean(axis=0) for k in range(n_clusters)])
    
    # Calcular custo e atualizar theta
    current_cost = cost(theta, new_centroids)
    theta = opt.step(cost, theta, new_centroids)
    
    # Critério de parada
    if np.linalg.norm(new_centroids - centroids) < 1e-4:
        break
    centroids = new_centroids
    
    print(f"Iteração {it+1}, Custo: {current_cost:.4f}")

# Resultados finais
print("Centróides finais:\n", centroids)