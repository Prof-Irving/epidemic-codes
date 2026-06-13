import numpy as np
from tqdm import tqdm
import numba as nb
from numba_progress import ProgressBar, ProgressBarType

class SEIRD():

    # 'S' -> 0 Suscuptible
    # 'E' -> 1 Exposed
    # 'I' -> 2 Infective
    # 'R' -> 3 Removed (Immune)
    # 'D' -> 4 Dead
    # 'n' -> 5 Null (Removed by death)
    # 'ID' -> 6 Infected and Dead (See "Store types of infection")

    def __init__(self, S_i, E_i, I_i, R_i, D_i,
               beta_I, beta_D, T_E, T_I, T_D, f, dt):
        # Create object properties
        self.N = S_i + E_i + I_i + R_i
        self.beta_I = beta_I
        self.beta_D = beta_D
        self.T_E = T_E
        self.T_I = T_I
        self.T_D = T_D
        self.f = f
        self.dt = dt

        # Calculate constant change probabilities
        self.p_EI = dt/T_E
        self.p_IR = dt * (1-f)/T_I
        self.p_ID = dt * f/T_I
        self.p_D = dt/T_D

        # Create list with groups values
        self.S_serie = [S_i]
        self.E_serie = [E_i]
        self.I_serie = [I_i]
        self.R_serie = [R_i]
        self.D_serie = [D_i]
        self.t_serie = [0.0]

        # Create agents
        state = np.array([0]*S_i + [1]*E_i + [2]*I_i + [3]*R_i + [4]*D_i)
        self.agents = state

        # Create 
        self.infection_type = np.zeros(self.N)

    def evolve(self):

        # Create temporary array
        agents_temp = self.agents

        # Get index of agents in different groups
        indexes = (np.where(agents_temp == i)[0] for i in [0,1,2,4])
        sus_index, exp_index, inf_index, dea_index = indexes

        # Change suseptible into infected
        # Calculate change probability
        p_SI_I = self.beta_I * len(inf_index)/self.N * self.dt
        p_SI_D = self.beta_D * len(dea_index)/self.N * self.dt
        # Verify stochastic changes
        SI_I_condition = np.random.uniform(0, 1, len(sus_index)) <= p_SI_I
        SI_D_condition = np.random.uniform(0, 1, len(sus_index)) <= p_SI_D
        # Some 'S' have contact with 'I' and 'D'
        change_condition = SI_I_condition | SI_D_condition
        # Change and store states in temporary array
        suscepted_exposed = np.where(change_condition, 1, 0) 
        agents_temp[sus_index] = suscepted_exposed

        # Store types of infection
        type_temp = self.infection_type[sus_index]
        type_temp = np.where(SI_I_condition, 2, type_temp)
        type_temp = np.where(SI_D_condition, 4, type_temp)
        type_temp = np.where(SI_I_condition & SI_D_condition, 6, type_temp)
        self.infection_type[sus_index] = type_temp

        # Change exposed into infected
        change_condition = np.random.uniform(0, 1, len(exp_index)) <= self.p_EI
        exposed_infected = np.where(change_condition, 2, 1)
        agents_temp[exp_index] = exposed_infected

        # Change infected into removed and dead
        x = np.random.uniform(0,1,len(inf_index))
        cond_list = [x <= self.p_IR, x <= self.p_IR + self.p_ID]
        infected_removed_dead = np.where(cond_list[0], 3, np.where(cond_list[1], 4, 2))
        agents_temp[inf_index] = infected_removed_dead

        # Change dead into null
        change_condition = np.random.uniform(0, 1, len(dea_index)) <= self.p_D
        infected_dead = np.where(change_condition, 5, 4)
        agents_temp[dea_index] = infected_dead

        # Update agents state
        self.agents = agents_temp


    def step(self, steps):

        for _ in tqdm(range(steps)):

            self.evolve()
            
            states = self.agents

            equal_condition = states[None,:] == np.array([0,1,2,3,4])[:,None]
            S, E, I, R, D = np.sum(equal_condition, axis=1)
            #S, E, I, R, D = (np.sum(states == i) for i in [0,1,2,3,4])

            self.S_serie.append(S)
            self.E_serie.append(E)
            self.I_serie.append(I)
            self.R_serie.append(R)
            self.D_serie.append(D)
            self.t_serie.append(self.t_serie[-1]+self.dt)

            # Recalculate alive population
            self.N = S+E+I+R


    def get_series(self):

        tt = np.asarray(self.t_serie)
        SS = np.asarray(self.S_serie)
        EE = np.asarray(self.E_serie)
        II = np.asarray(self.I_serie)
        RR = np.asarray(self.R_serie)
        DD = np.asarray(self.D_serie)

        NN = SS + EE + II + RR

        '''
        SS = SS/NN
        EE = EE/NN
        II = II/NN
        RR = RR/NN
        DD = DD/NN
        '''

        return tt, SS, EE, II, RR, DD, NN


spec = [('N', nb.i8), ('S_i', nb.i8), ('E_i', nb.i8), ('I_i', nb.i8), ('R_i', nb.i8), ('D_i', nb.i8),
        ('beta_I', nb.f8), ('beta_D', nb.f8), ('T_E', nb.f8), ('T_I', nb.f8), ('T_D', nb.f8),
        ('f', nb.f8), ('dt', nb.f8),
        ('p_EI', nb.f8), ('p_IR', nb.f8), ('p_ID', nb.f8), ('p_D', nb.f8),
        ('S_serie', nb.types.ListType(nb.types.int64)),
        ('E_serie', nb.types.ListType(nb.types.int64)),
        ('I_serie', nb.types.ListType(nb.types.int64)),
        ('R_serie', nb.types.ListType(nb.types.int64)),
        ('D_serie', nb.types.ListType(nb.types.int64)),
        ('t_serie', nb.types.ListType(nb.types.float64)),
        ('agents', nb.i8[:]), ('infection_type', nb.i8[:])]

@nb.experimental.jitclass(spec)
class SEIRD_numba():

    # 'S' -> 0 Suscuptible
    # 'E' -> 1 Exposed
    # 'I' -> 2 Infective
    # 'R' -> 3 Removed (Immune)
    # 'D' -> 4 Dead
    # 'n' -> 5 Null (Removed by death)
    # 'ID' -> 6 Infected and Dead (See "Store types of infection")

    def __init__(self, S_i, E_i, I_i, R_i, D_i,
               beta_I, beta_D, T_E, T_I, T_D, f, dt):
        # Create object properties
        self.N = S_i + E_i + I_i + R_i
        self.beta_I = beta_I
        self.beta_D = beta_D
        self.T_E = T_E
        self.T_I = T_I
        self.T_D = T_D
        self.f = f
        self.dt = dt

        # Calculate constant change probabilities
        self.p_EI = dt/T_E
        self.p_IR = dt * (1-f)/T_I
        self.p_ID = dt * f/T_I
        self.p_D = dt/T_D

        # Create list with groups values
        self.S_serie = nb.typed.List([S_i])
        self.E_serie = nb.typed.List([E_i])
        self.I_serie = nb.typed.List([I_i])
        self.R_serie = nb.typed.List([R_i])
        self.D_serie = nb.typed.List([D_i])
        self.t_serie = nb.typed.List([0.0])

        # Create agents
        state = np.array([0]*S_i + [1]*E_i + [2]*I_i + [3]*R_i + [4]*D_i)
        self.agents = state

        # Create 
        self.infection_type = np.zeros(self.N, dtype=np.int64)

    def evolve(self):

        # Create temporary array
        agents_temp = self.agents

        # Get index of agents in different groups
        sus_index = np.where(agents_temp == 0)[0]
        exp_index = np.where(agents_temp == 1)[0]
        inf_index = np.where(agents_temp == 2)[0]
        dea_index = np.where(agents_temp == 4)[0]

        # Change suseptible into infected
        # Calculate change probability
        p_SI_I = self.beta_I * len(inf_index)/self.N * self.dt
        p_SI_D = self.beta_D * len(dea_index)/self.N * self.dt
        # Verify stochastic changes
        SI_I_condition = np.random.uniform(0, 1, len(sus_index)) <= p_SI_I
        SI_D_condition = np.random.uniform(0, 1, len(sus_index)) <= p_SI_D
        # Some 'S' have contact with 'I' and 'D'
        change_condition = SI_I_condition | SI_D_condition
        # Change and store states in temporary array
        suscepted_exposed = np.where(change_condition, 1, 0) 
        agents_temp[sus_index] = suscepted_exposed

        # Store types of infection
        type_temp = self.infection_type[sus_index]
        type_temp = np.where(SI_I_condition, 2, type_temp)
        type_temp = np.where(SI_D_condition, 4, type_temp)
        type_temp = np.where(SI_I_condition & SI_D_condition, 6, type_temp)
        self.infection_type[sus_index] = type_temp

        # Change exposed into infected
        change_condition = np.random.uniform(0, 1, len(exp_index)) <= self.p_EI
        exposed_infected = np.where(change_condition, 2, 1)
        agents_temp[exp_index] = exposed_infected

        # Change infected into removed and dead
        x = np.random.uniform(0,1,len(inf_index))
        cond_list = [x <= self.p_IR, x <= self.p_IR + self.p_ID]
        infected_removed_dead = np.where(cond_list[0], 3, np.where(cond_list[1], 4, 2))
        agents_temp[inf_index] = infected_removed_dead

        # Change dead into null
        change_condition = np.random.uniform(0, 1, len(dea_index)) <= self.p_D
        infected_dead = np.where(change_condition, 5, 4)
        agents_temp[dea_index] = infected_dead

        # Update agents state
        self.agents = agents_temp


    def step(self, steps):

        for _ in range(steps):

            self.evolve()
            
            states = self.agents

            equal_condition = states[None,:] == np.array([0,1,2,3,4])[:,None]
            S, E, I, R, D = np.sum(equal_condition, axis=1)
            #S, E, I, R, D = (np.sum(states == i) for i in [0,1,2,3,4])

            self.S_serie.append(S)
            self.E_serie.append(E)
            self.I_serie.append(I)
            self.R_serie.append(R)
            self.D_serie.append(D)
            self.t_serie.append(self.t_serie[-1]+self.dt)

            # Recalculate alive population
            self.N = S+E+I+R


    def get_series(self):

        tt = np.asarray(self.t_serie)
        SS = np.asarray(self.S_serie)
        EE = np.asarray(self.E_serie)
        II = np.asarray(self.I_serie)
        RR = np.asarray(self.R_serie)
        DD = np.asarray(self.D_serie)

        NN = SS + EE + II + RR

        '''
        SS = SS/NN
        EE = EE/NN
        II = II/NN
        RR = RR/NN
        DD = DD/NN
        '''

        return tt, SS, EE, II, RR, DD, NN

        
signature = nb.types.Tuple(
                            (nb.f8[:,:], nb.f8[:,:,:])
                            )(
                            nb.i8[:], nb.f8[:,:], nb.f8, nb.f8, ProgressBarType
                            )
@nb.jit(signature, nopython = True, parallel = True, nogil = True)
def func_sim_agents_numba(init_conditions, params, dt, tf, progress_proxy):

    N_states = len(init_conditions)
    N_sims = len(params)
    N_steps = int(tf/dt + 1)

    series = np.empty((N_states, N_sims, N_steps), dtype=np.float64)

    for i in nb.prange(N_sims):

        epidemic = SEIRD_numba(init_conditions[0], init_conditions[1], init_conditions[2], init_conditions[3], init_conditions[4], 
                                params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], dt)

        epidemic.step(int(tf/dt))

        # Coletar arrays da simulação (excluindo array do tempo) e adicionar na listata
        all_series = epidemic.get_series()[1:]

        for j in range(N_states):
            series[j,i,:] = all_series[j]

        progress_proxy.update(1)

    return params, series 

def func_sim_agents_numba_progressbar(init_conditions, params, dt, tf):
    num_iterations = len(params)
    with ProgressBar(total=num_iterations) as progress:
       params, series = func_sim_agents_numba(init_conditions, params, dt, tf, progress)
       return params, *series