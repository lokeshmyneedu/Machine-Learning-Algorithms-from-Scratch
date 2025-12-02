graph TD
    Start([Input: Dataset N rows, D cols]) --> IsLinear{Is Relationship<br/>Linear?}

    %% --- LINEAR BRANCH ---
    IsLinear -- Yes --> Regularization{Regularization<br/>Check}
    
    Regularization -- "Multicollinearity / High Variance" --> Ridge[<b>Ridge Regression</b><br/>L2 Penalty]
    Regularization -- "Need Feature Selection" --> Lasso[<b>Lasso Regression</b><br/>L1 Penalty]
    Regularization -- "Correlated & Sparse" --> Elastic[<b>ElasticNet</b><br/>L1 + L2 Penalty]
    Regularization -- "Simple / Baseline" --> OLS[<b>OLS Linear</b><br/>No Penalty]

    Ridge --> SolverCheck{Dataset Size?}
    Lasso --> CoordDesc[<b>Solver:</b><br/>Coordinate Descent]
    Elastic --> CoordDesc
    OLS --> SolverCheck

    SolverCheck -- "Small (< 10k rows)" --> NormalEq[<b>Normal Equation</b><br/>Exact Solution<br/>(Primal)]
    SolverCheck -- "Large (> 10k rows)" --> SGD_Lin[<b>SGD Optimizer</b><br/>Approximate<br/>(Primal)]

    %% --- NON-LINEAR BRANCH ---
    IsLinear -- No --> CurveType{Curve Complexity?}

    CurveType -- "Simple Curve / Low D" --> Poly[<b>Polynomial Features</b>]
    Poly --> Regularization

    CurveType -- "Complex / Disjointed" --> StreamingCheck{Streaming Data?}

    StreamingCheck -- "Yes (Real-time)" --> KernelSGD[<b>Kernel SGD</b><br/>Functional Gradient Descent<br/>(Dual w/ Budget)]

    StreamingCheck -- "No (Batch)" --> BatchSize{Batch Size?}
    
    BatchSize -- "Small (< 20k rows)" --> KRR[<b>Kernel Ridge Regression</b><br/>Exact Dual Solution<br/>(Dual)]
    
    BatchSize -- "Massive (> 100k rows)" --> RFF[<b>RFF + Linear SGD</b><br/>Random Fourier Features<br/>(Primal Approx)]