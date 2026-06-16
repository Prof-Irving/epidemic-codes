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


class SEIRD_II():

    # 0 -> S Susceptible
    # 1 -> E Exposed
    # 2 -> I Infective
    # 3 -> II Isolated Infective
    # 4 -> R Removed (Immune)
    # 5 -> D Dead
    # 6 -> n Null (Removed by death)
    # 7 -> ID Infected and Dead (See "Store types of infection")

    def __init__(self, S_i, E_i, I_i, II_i, R_i, D_i,
               beta_I, beta_II, beta_D, T_E, T_I, alpha, T_D, f, dt):
        # Create object properties
        self.N_i = S_i + E_i + I_i + R_i
        self.N = S_i + E_i + I_i + R_i
        self.beta_I = beta_I
        self.beta_II = beta_II
        self.beta_D = beta_D
        self.T_E = T_E
        self.T_I = T_I
        self.T_D = T_D
        self.alpha = alpha
        self.f = f
        self.dt = dt

        # Calculate constant change probabilities
        self.p_EI = dt/T_E
        self.p_III = dt*alpha
        self.p_IR = dt * (1-f)/T_I
        self.p_ID = dt * f/T_I
        self.p_D = dt/T_D

        # Calculate constant change probabilities per contact (p = 0.01)
        p = 0.01
        self.pc = p * self.dt
        # How many contacts one I, II and D agent can do
        self.c_I = beta_I/p
        self.c_II = beta_II/p
        self.c_D = beta_D/p
    
        # Create list with groups values
        self.S_serie = [S_i]
        self.E_serie = [E_i]
        self.I_serie = [I_i]
        self.II_serie = [II_i]
        self.R_serie = [R_i]
        self.D_serie = [D_i]
        self.t_serie = [0.0]

        # Create agents
        state = np.array([0]*S_i + [1]*E_i + [2]*I_i + [3]*II_i + [4]*R_i + [5]*D_i)
        self.agents = state

        # Create
        self.infection_type = np.zeros(self.N)

    def evolve(self):

        # Create temporary array
        agents_temp = self.agents

        # Get index of agents in different groups
        indexes = (np.where(agents_temp == i)[0] for i in [0,1,2,3,5])
        sus_index, exp_index, inf_index, isol_inf_index, dea_index = indexes

        # Change suseptible into infected
        # Verify stochastic infections
        S_frac = self.S_serie[-1]/self.N # Fraction of S in population
        N_frac = self.N/self.N_i # Population size correction factor
        c_I_new = self.c_I * N_frac # Contact number correction
        c_II_new = self.c_II * N_frac
        c_D_new = self.c_D * N_frac
        total_contacts_I = self.I_serie[-1] * round(c_I_new * S_frac)
        total_contacts_II = self.II_serie[-1] * round(c_II_new * S_frac)
        total_contacts_D = self.D_serie[-1] * round(c_D_new * S_frac)
        total_contacts = total_contacts_I + total_contacts_II + total_contacts_D
        #change_condition_I = np.random.rand(total_contacts_I) <= self.pc
        #change_condition_II = np.random.rand(total_contacts_II) <= self.pc
        #change_condition_D = np.random.rand(total_contacts_D) <= self.pc
        #change_condition = np.concatenate((change_condition_I, change_condition_II, change_condition_D))
        change_condition = np.random.rand(total_contacts) <= self.pc
        # Select random susceptible agents to have contacts
        S_selected = np.random.randint(0, self.S_serie[-1], total_contacts)
        S_changed = np.unique(S_selected[change_condition])
        # Change temporary array
        agents_temp[sus_index[S_changed]] = 1

        # Change exposed into infected
        change_condition = np.random.rand(self.E_serie[-1]) <= self.p_EI
        agents_temp[exp_index[change_condition]] = 2

        # Change infected into isolated infected, removed and dead
        x = np.random.rand(self.I_serie[-1])
        cond_list = [x <= self.p_III,
                        x <= self.p_III + self.p_IR,
                        x <= self.p_III + self.p_IR + self.p_ID]
        agents_temp[inf_index[cond_list[2]]] = 5
        agents_temp[inf_index[cond_list[1]]] = 4
        agents_temp[inf_index[cond_list[0]]] = 3

        # Change isolated infected into removed and null (dead removed)
        # This means the isolated deads will not infect
        x = np.random.rand(self.II_serie[-1])
        cond_list = [x <= self.p_IR, 
                        x <= self.p_IR + self.p_ID]
        agents_temp[isol_inf_index[cond_list[1]]] = 6
        agents_temp[isol_inf_index[cond_list[0]]] = 4

        # Change dead into null
        change_condition = np.random.rand(self.D_serie[-1]) <= self.p_D
        agents_temp[dea_index[change_condition]] = 6

        # Update agents state
        self.agents = agents_temp


    def step(self, steps):

        for _ in tqdm(range(steps)):

            self.evolve()

            states = self.agents
            
            S, E, I, II, R, D = (np.sum(states == i) for i in [0,1,2,3,4,5])

            self.S_serie.append(S)
            self.E_serie.append(E)
            self.I_serie.append(I)
            self.II_serie.append(II)
            self.R_serie.append(R)
            self.D_serie.append(D)
            self.t_serie.append(self.t_serie[-1]+self.dt)

            # Recalculate alive population
            self.N = S+E+I+II+R


    def get_series(self):

        t = np.asarray(self.t_serie)
        S = np.asarray(self.S_serie)
        E = np.asarray(self.E_serie)
        I = np.asarray(self.I_serie)
        II = np.asarray(self.II_serie)
        R = np.asarray(self.R_serie)
        D = np.asarray(self.D_serie)

        N = S + E + I + II + R

        '''
        S = S/N
        E = E/N
        I = I/N
        II = II/N
        R = R/N
        D = D/N
        '''

        return t, S, E, I, II, R, D, N


spec = [('N', nb.i8), ('S_i', nb.i8), ('E_i', nb.i8), ('I_i', nb.i8), ('II_i', nb.i8), ('R_i', nb.i8), ('D_i', nb.i8),
        ('beta_I', nb.f8), ('beta_II', nb.f8), ('beta_D', nb.f8), ('T_E', nb.f8), ('alpha', nb.f8), ('T_I', nb.f8), ('T_D', nb.f8),
        ('f', nb.f8), ('dt', nb.f8),
        ('p_EI', nb.f8), ('p_III', nb.f8), ('p_IR', nb.f8), ('p_ID', nb.f8), ('p_D', nb.f8), ('pc', nb.f8),
        ('c_I', nb.f8), ('c_II', nb.f8), ('c_D', nb.f8),
        ('S_serie', nb.types.ListType(nb.types.int64)),
        ('E_serie', nb.types.ListType(nb.types.int64)),
        ('I_serie', nb.types.ListType(nb.types.int64)),
        ('II_serie', nb.types.ListType(nb.types.int64)),
        ('R_serie', nb.types.ListType(nb.types.int64)),
        ('D_serie', nb.types.ListType(nb.types.int64)),
        ('t_serie', nb.types.ListType(nb.types.float64)),
        ('agents', nb.i8[:]), ('infection_type', nb.i8[:])]

@nb.experimental.jitclass(spec)
class SEIRD_II_numba():

    # 0 -> S Susceptible
    # 1 -> E Exposed
    # 2 -> I Infective
    # 3 -> II Isolated Infective
    # 4 -> R Removed (Immune)
    # 5 -> D Dead
    # 6 -> n Null (Removed by death)
    # 7 -> ID Infected and Dead (See "Store types of infection")

    def __init__(self, S_i, E_i, I_i, II_i, R_i, D_i,
               beta_I, beta_II, beta_D, T_E, T_I, alpha, T_D, f, dt):
        # Create object properties
        self.N_i = S_i + E_i + I_i + R_i
        self.N = S_i + E_i + I_i + R_i
        self.beta_I = beta_I
        self.beta_II = beta_II
        self.beta_D = beta_D
        self.T_E = T_E
        self.T_I = T_I
        self.T_D = T_D
        self.alpha = alpha
        self.f = f
        self.dt = dt

        # Calculate constant change probabilities
        self.p_EI = dt/T_E
        self.p_III = dt*alpha
        self.p_IR = dt * (1-f)/T_I
        self.p_ID = dt * f/T_I
        self.p_D = dt/T_D

        # Calculate constant change probabilities per contact (p = 0.01)
        p = 0.01
        self.pc = p * self.dt
        # How many contacts one I, II and D agent can do
        self.c_I = beta_I/p
        self.c_II = beta_II/p
        self.c_D = beta_D/p
    
        # Create list with groups values
        self.S_serie = nb.typed.List([S_i])
        self.E_serie = nb.typed.List([E_i])
        self.I_serie = nb.typed.List([I_i])
        self.II_serie = nb.typed.List([II_i])
        self.R_serie = nb.typed.List([R_i])
        self.D_serie = nb.typed.List([D_i])
        self.t_serie = nb.typed.List([0.0])

        # Create agents
        state = np.array([0]*S_i + [1]*E_i + [2]*I_i + [3]*II_i + [4]*R_i + [5]*D_i)
        self.agents = state

        # Create
        self.infection_type = np.zeros(self.N)

    def evolve(self):

        # Create temporary array
        agents_temp = self.agents

        # Get index of agents in different groups
        sus_index = np.where(agents_temp == 0)[0]
        exp_index = np.where(agents_temp == 1)[0]
        inf_index = np.where(agents_temp == 2)[0]
        isol_inf_index = np.where(agents_temp == 3)[0]
        dea_index = np.where(agents_temp == 5)[0]

        # Change suseptible into infected
        # Verify stochastic infections
        S_frac = self.S_serie[-1]/self.N # Fraction of S in population
        N_frac = self.N/self.N_i # Population size correction factor
        c_I_new = self.c_I * N_frac # Contact number correction
        c_II_new = self.c_II * N_frac
        c_D_new = self.c_D * N_frac
        total_contacts_I = self.I_serie[-1] * round(c_I_new * S_frac)
        total_contacts_II = self.II_serie[-1] * round(c_II_new * S_frac)
        total_contacts_D = self.D_serie[-1] * round(c_D_new * S_frac)
        total_contacts = total_contacts_I + total_contacts_II + total_contacts_D
        #change_condition_I = np.random.rand(total_contacts_I) <= self.pc
        #change_condition_II = np.random.rand(total_contacts_II) <= self.pc
        #change_condition_D = np.random.rand(total_contacts_D) <= self.pc
        #change_condition = np.concatenate((change_condition_I, change_condition_II, change_condition_D))
        change_condition = np.random.rand(total_contacts) <= self.pc
        # Select random susceptible agents to have contacts
        S_selected = np.random.randint(0, self.S_serie[-1], total_contacts)
        S_changed = np.unique(S_selected[change_condition])
        # Change temporary array
        agents_temp[sus_index[S_changed]] = 1

        # Change exposed into infected
        change_condition = np.random.rand(self.E_serie[-1]) <= self.p_EI
        agents_temp[exp_index[change_condition]] = 2

        # Change infected into isolated infected, removed and dead
        x = np.random.rand(self.I_serie[-1])
        cond_list = [x <= self.p_III,
                        x <= self.p_III + self.p_IR,
                        x <= self.p_III + self.p_IR + self.p_ID]
        agents_temp[inf_index[cond_list[2]]] = 5
        agents_temp[inf_index[cond_list[1]]] = 4
        agents_temp[inf_index[cond_list[0]]] = 3

        # Change isolated infected into removed and null (dead removed)
        # This means the isolated deads will not infect
        x = np.random.rand(self.II_serie[-1])
        cond_list = [x <= self.p_IR, 
                        x <= self.p_IR + self.p_ID]
        agents_temp[isol_inf_index[cond_list[1]]] = 6
        agents_temp[isol_inf_index[cond_list[0]]] = 4

        # Change dead into null
        change_condition = np.random.rand(self.D_serie[-1]) <= self.p_D
        agents_temp[dea_index[change_condition]] = 6

        # Update agents state
        self.agents = agents_temp


    def step(self, steps):

        for _ in range(steps):

            self.evolve()

            states = self.agents
            
            equal_condition = states[None,:] == np.array([0,1,2,3,4,5])[:,None]
            S, E, I, II, R, D = np.sum(equal_condition, axis=1)

            self.S_serie.append(S)
            self.E_serie.append(E)
            self.I_serie.append(I)
            self.II_serie.append(II)
            self.R_serie.append(R)
            self.D_serie.append(D)
            self.t_serie.append(self.t_serie[-1]+self.dt)

            # Recalculate alive population
            self.N = S+E+I+II+R


    def get_series(self):

        t = np.asarray(self.t_serie)
        S = np.asarray(self.S_serie)
        E = np.asarray(self.E_serie)
        I = np.asarray(self.I_serie)
        II = np.asarray(self.II_serie)
        R = np.asarray(self.R_serie)
        D = np.asarray(self.D_serie)

        N = S + E + I + II + R

        '''
        S = S/N
        E = E/N
        I = I/N
        II = II/N
        R = R/N
        D = D/N
        '''

        return t, S, E, I, II, R, D, N


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

'''
signature = nb.types.Tuple(
                            (nb.f8[:,:], nb.f8[:,:,:])
                            )(
                            nb.types.unicode_type, nb.i8[:], nb.f8[:,:], nb.f8, nb.f8, ProgressBarType
                            )
@nb.jit(signature, nopython = True, parallel = True, nogil = True)
def func_sim_agents_numba(model, init_conditions, params, dt, tf, progress_proxy):

    N_states = len(init_conditions)
    N_sims = len(params)
    N_steps = int(tf/dt + 1)

    series = np.empty((N_states, N_sims, N_steps), dtype=np.float64)

    for i in nb.prange(N_sims):

        match model:

            case 'SEIRD':
                epidemic = SEIRD(init_conditions[0], init_conditions[1], init_conditions[2], init_conditions[3], init_conditions[4], 
                                    params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], dt)

            case 'SEIRD_numba':
                epidemic = SEIRD_numba(init_conditions[0], init_conditions[1], init_conditions[2], init_conditions[3], init_conditions[4], 
                                    params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], dt)
            
            case 'SEIRD_II':
                epidemic = SEIRD_II(init_conditions[0], init_conditions[1], init_conditions[2], init_conditions[3], init_conditions[4], init_conditions[5], 
                                    params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][6], params[i][7], dt)

            case 'SEIRD_II_numba':
                epidemic = SEIRD_II_numba(init_conditions[0], init_conditions[1], init_conditions[2], init_conditions[3], init_conditions[4], init_conditions[5], 
                                    params[i][0], params[i][1], params[i][2], params[i][3], params[i][4], params[i][5], params[i][6], params[i][7], dt)
        
        epidemic.step(int(tf/dt))

        # Coletar arrays da simulação (excluindo array do tempo) e adicionar na listata
        all_series = epidemic.get_series()[1:]

        for j in range(N_states):
            series[j,i,:] = all_series[j]

        progress_proxy.update(1)

    return params, series 

def func_sim_agents_numba_progressbar(model, init_conditions, params, dt, tf):
    num_iterations = len(params)
    with ProgressBar(total=num_iterations) as progress:
       params, series = func_sim_agents_numba(model, init_conditions, params, dt, tf, progress)
       return params, *series
'''