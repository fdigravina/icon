:- dynamic best_path/2.

main :-
    
    retractall(bound(_)),
    retractall(best_path(_, _)), % Pulizia dei fatti precedenti
    
    consult('algoritmi scartati/branch and bound/Prolog/input.txt'),
    setof(Node, X^Y^edge(Node, X, Y), Nodes),

    %write('Nodi: '), write(Nodes), nl,

    %bound(Bound),
    %write('bound: '), write(Bound), nl, nl, nl,

    Start = 1,
    exclude(==(Start), Nodes, RestNodes),
    Path = [Start],
    
    % Inizia la ricerca del percorso migliore
    tsp(Start, RestNodes, Path, 0, Start),
    
    best_path(BestCost, BestPath),
    
    write('Best Tour: '), write(BestPath), nl,
    write('Costo: '), write(BestCost), nl.

% Caso base: tutti i nodi visitati, ritorna al nodo iniziale e calcola il costo
tsp(Current, [], Path, Cost, Start) :-
    edge(Current, Start, CostToStart),
    append(Path, [Start], PathToStart),
    CostTot is Cost + CostToStart,
    update_best_path(PathToStart, CostTot).

% Caso ricorsivo: trova il percorso migliore tra i nodi rimanenti
tsp(Current, Nodes, Path, CostP, Start) :-
    %write('Current: '), write(Current), nl,
    %write('Nodes: '), write(Nodes), nl,
    %write('Path: '), write(Path), nl,
    %write('Cost: '), write(CostP), nl, nl, nl,
    select(Next, Nodes, RestNodes),
    edge(Current, Next, Cost),
    bound(Bound),
    CostTot is Cost + CostP,
    CostTot =< Bound,
    append(Path, [Next], NewPath),
    tsp(Next, RestNodes, NewPath, CostTot, Start),
    fail. % Forza il backtracking per esplorare tutte le possibilità

update_best_path(NewPath, NewCost) :-
    
    bound(CurrentBound),
    
    %write('Path: '), write(NewPath), nl,
    %write('Cost: '), write(NewCost), nl,
    %write('Bound: '), write(CurrentBound), nl,
    
    NewCost < CurrentBound,
    
    retractall(best_path(_, _)),
    write('Path: '), write(NewPath), nl,
    write('Cost: '), write(NewCost), nl,
    assertz(best_path(NewCost, NewPath)),
    
    retractall(bound(_)),
    assertz(bound(NewCost)),  % Aggiornare il bound
    
    best_path(Cost, Path),
    bound(Bound),
    
    write('Percorso: '), write(Path), nl,
    write('Costo: '), write(Cost), nl,
    write('Nuovo bound: '), write(Bound), nl.