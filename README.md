# LabPhysicsTools

LabPhysicsTools è una libreria Python sviluppata per semplificare, standardizzare e automatizzare le analisi statistiche e la resa grafica nei laboratori di fisica sperimentale. Il pacchetto racchiude routine complete di regressione non lineare basate sulla minimizzazione del chi-quadro, calcolo dei profili di confidenza e pipeline di plotting ottimizzate per report tecnici e documenti LaTeX.

## Caratteristiche Principali

1. Regressione $\chi^2$ e Metodi ODR (chi_squared_fit): routine di fit non lineare con stima rigorosa degli errori sui parametri ottimali, supporto per incertezze su entrambi gli assi tramite regressione a distanza ortogonale (ODR), calcolo automatico di $\chi^2$, gradi di libertà ($ndf$), p-value e matrici di covarianza, algoritmi di recupero per la stima iniziale dei parametri (global guess parameter recovery).
3. Mappatura 2D del $\chi^2$ ed Ellissoidi di Confidenza (chi_squared_2d_map): generazione di mappe bidimensionali di $\chi^2$ nello spazio dei parametri per individuare correlazioni e minimi locali, tracciamento diretto dei contorni di confidenza a $1\sigma$, $2\sigma$ e $3\sigma$ ($\Delta\chi^2$), esportazione di grafici diagnostici per la verifica dell'approssimazione parabolica attorno al minimo.
6. Tipografia e Visualizzazione Scientifica (plot_typo): layout coerenti per figure a singolo e doppio pannello (es. dato sperimentale + pannello dei residui normalizzati), gestione avanzata delle scale lineari e logaritmiche con spaziatura e griglie ad alta leggibilità, configurazione tipografica integrata con gli standard di impaginazione accademica (font sizing, etichette asse e formattazione LaTeX-ready).   

Repository: https://github.com/el-coda-coda/LabPhysicsTools.git
