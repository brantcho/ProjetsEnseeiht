
@doc doc"""
#### Objet
Cette fonction implémente l'algorithme de Newton pour résoudre un problème d'optimisation sans contraintes

#### Syntaxe
```julia
xmin,fmin,flag,nb_iters = Algorithme_de_Newton(f,gradf,hessf,x0,option)
```

#### Entrées :
   - f       : (Function) la fonction à minimiser
   - gradf   : (Function) le gradient de la fonction f
   - hessf   : (Function) la hessienne de la fonction f
   - x0      : (Array{Float,1}) première approximation de la solution cherchée
   - options : (Array{Float,1})
       - max_iter      : le nombre maximal d'iterations
       - Tol_abs       : la tolérence absolue
       - Tol_rel       : la tolérence relative
       - epsilon       : epsilon pour les tests de stagnation

#### Sorties:
   - xmin    : (Array{Float,1}) une approximation de la solution du problème  : ``\min_{x \in \mathbb{R}^{n}} f(x)``
   - fmin    : (Float) ``f(x_{min})``
   - flag    : (Integer) indique le critère sur lequel le programme s'est arrêté (en respectant cet ordre de priorité si plusieurs critères sont vérifiés)
      - 0    : CN1 
      - 1    : stagnation du xk
      - 2    : stagnation du f
      - 3    : nombre maximal d'itération dépassé
   - nb_iters : (Integer) le nombre d'itérations faites par le programme

#### Exemple d'appel
```@example
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
x0 = [1; 0]
options = []
xmin,fmin,flag,nb_iters = Algorithme_De_Newton(f,gradf,hessf,x0,options)
```
"""
function Algorithme_De_Newton(f::Function,gradf::Function,hessf::Function,x0,options)

    "# Si options == [] on prends les paramètres par défaut"
    if options == []
        max_iter = 100
        Tol_abs = sqrt(eps())
        Tol_rel = 1e-15
        epsilon = 1.e-2
    else
        max_iter = options[1]
        Tol_abs = options[2]
        Tol_rel = options[3]
        epsilon = options[4]
    end
        xmin = x0
        fmin = f(x0) 
        flag = -1  
        nb_iters = 0 
    
    
    if  norm(gradf(x0)) <= max(Tol_abs , Tol_rel*norm(gradf(x0)))
            flag = 0
    end 
         
    while ( flag == -1 )
       x1 = xmin
       f1 = f(xmin) 
       d = hessf(xmin) \ (- gradf(xmin)) 
       xmin = xmin + d 
       nb_iters = nb_iters + 1 
       fmin = f(xmin) 
     
      
       if  norm(gradf(xmin)) <= max(Tol_abs , Tol_rel*norm(gradf(x0)))
            flag = 0
         
       elseif norm(xmin - x1) <= epsilon*max(Tol_abs , Tol_rel*norm(x1)) 
            flag = 1
       
       elseif abs(fmin - f1) <= epsilon*max(Tol_abs , Tol_rel*norm(f1))
           flag = 2
            
       elseif  (nb_iters == max_iter )
           flag = 3    
       else 
           flag = -1      
           
       end 
                
    end
        
        
    return xmin,fmin,flag,nb_iters
end
