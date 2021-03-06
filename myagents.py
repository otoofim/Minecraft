"""
 Artificial Inteligence (H)
 Assessed Exercise 2017/2018

 Team 12
 Jack Moulson           (Reg 2326573)
 Mohammed Otoofi        (Reg 2263537)
 Niall Morley           (Reg 2356240)

 Tested with Python 2.7
"""

#----------------------------------------------------------------------------------------------------------------#
class SolutionReport(object):
    # Please do not modify this custom class!
    def __init__(self):
        """ Constructor for the solution info
        class which is used to store a summary of
        the mission and agent performance """

        self.start_datetime_wallclock = None;
        self.end_datetime_wallclock = None;
        self.action_count = 0;
        self.reward_cumulative = 0;
        self.mission_xml = None;
        self.mission_type = None;
        self.mission_seed = None;
        self.student_guid = None;
        self.mission_xml_as_expected = None;
        self.is_goal = None;
        self.is_timeout = None;

    def start(self):
        self.start_datetime_wallclock = datetime.datetime.now()

    def stop(self):
        self.checkGoal()
        self.end_datetime_wallclock = datetime.datetime.now()

    def setMissionXML(self, mission_xml):
        self.mission_xml = mission_xml

    def setMissionType(self, mission_type):
        self.mission_type = mission_type

    def setMissionSeed(self, mission_seed):
        self.mission_seed = mission_seed

    def setStudentGuid(self, student_guid):
        self.student_guid = student_guid

    def addAction(self):
        self.action_count += 1

    def addReward(self, reward, datetime_):
        self.reward_cumulative += reward

    def checkMissionXML(self):
          """ This function could potentially part of the final check"""
          #-- This is not included currently for simplicity --#
          self.mission_xml_as_expected = 'Unknown'

    def checkGoal(self):
          """ This function checks if the goal has been reached based on the expected reward structure (will not work if you change the mission xml!)"""
          #-- This is not included for simplicity --#
          if self.reward_cumulative!=None:
            x = round((abs(self.reward_cumulative)-abs(round(self.reward_cumulative)))*100);
            rem_goal = x%7
            rem_timeout = x%20
            if rem_goal==0 and x!=0:
                self.is_goal = True
            else:
                self.is_goal = False

            if rem_timeout==0 and x!=0:
                self.is_timeout = True
            else:
                self.is_timeout = False

#----------------------------------------------------------------------------------------------------------------#
class StateSpace(object):
    """ This is a datatype used to collect a number of important aspects of the environment
    It can be constructed online or be created offline using the Helper Agent """

    def __init__(self):
        # Constructor for the local state-space representation derived from the Orcale
        self.state_locations = None;
        self.state_actions = None;
        self.start_id = None;  # The id assigned to the start state
        self.goal_id = None;  # The id assigned to the initial goal state
        self.start_loc = None; # The real word coordinates of the start state
        self.goal_loc = None; # The real word coordinates of the goal state
        self.reward_states_n = None
        self.reward_states = None
        self.reward_sendcommand = None
        self.reward_timeout = None
        self.timeout = None


#----------------------------------------------------------------------------------------------------------------#
def GetMissionInstance( mission_type, mission_seed, agent_type):
    #Creates a specific instance of a given mission type """

    #Size of the problem/map
    msize = {
        'small': 10,
        'medium': 20,
        'large': 40,
    }

    # Timelimits
    mtimeout = {
        'small':   60000,
        'medium': 240000,
        'large':  960000,
        'helper': 200000,
    }

    # Number of intermediate rewards
    nir = {
        'small': 3,
        'medium': 9,
        'large': 27,
    }

    mission_type_tmp = mission_type
    if agent_type.lower()=='helper':
        mission_type_tmp = agent_type.lower()

    #-- Defines various parameters used in the generation of the mission.
    #-- It is crucial to scope these to the details of the mission, as it requires
    #-- knowledge of uncertainty/probability and random variables operating in the scenario
    random.seed(mission_seed)
    reward_goal = abs(round(random.gauss(1000, 400)))+0.0700
    reward_waypoint = round(abs(random.gauss(3, 15)))
    reward_timeout = -round(abs(random.gauss(1000, 400)))-0.2000
    reward_sendcommand = round(-random.randrange(2,10))

    n_intermediate_rewards = random.randrange(1,5) * nir.get(mission_type, 10) # How many intermediate rewards...?

    xml_str = '''<?xml version="1.0" encoding="UTF-8" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <About>
            <Summary>Mission for AI Class assessed exercise 2017/2018; University of Glasgow</Summary>
          </About>
          <ServerSection>
            <ServerInitialConditions>
              <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
              </Time>
              <Weather>clear</Weather>
              <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
              <MazeDecorator>
                <Seed>'''+str(mission_seed)+'''</Seed>
                <SizeAndPosition length="'''+str(msize.get(mission_type,100))+'''" width="'''+str(msize.get(mission_type,100))+'''" yOrigin="215" zOrigin="0" xOrigin="0" height="180"/>
                <GapProbability variance="0.3">0.2</GapProbability>
                <MaterialSeed>1</MaterialSeed>
                <AllowDiagonalMovement>false</AllowDiagonalMovement>
                <StartBlock fixedToEdge="true" type="emerald_block" height="1"/>
                <EndBlock fixedToEdge="false" type="redstone_block" height="12"/>
                <PathBlock type="glowstone" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="1"/>
                <FloorBlock type="air"/>
                <SubgoalBlock type="glowstone"/>
                <GapBlock type="stained_hardened_clay" colour="WHITE ORANGE MAGENTA LIGHT_BLUE YELLOW LIME PINK GRAY SILVER CYAN PURPLE BLUE BROWN GREEN RED BLACK" height="3"/>
                <Waypoints quantity="'''+str(n_intermediate_rewards)+'''">
                  <WaypointItem type="diamond_block"/>
                </Waypoints>
              </MazeDecorator>
              <ServerQuitFromTimeUp timeLimitMs="'''+str(mtimeout.get(mission_type_tmp,0))+'''" description="out_of_time"/>
              <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
          </ServerSection>
          <AgentSection>
            <Name>My Agent</Name>
            <AgentStart>
              <Placement x="0" y="216" z="90"/> <!-- will be overwritten by MazeDecorator -->
            </AgentStart>
            <AgentHandlers>
              <ObservationFromFullStats/>
              <ObservationFromRecentCommands/>
              <ObservationFromFullInventory/>
              <RewardForCollectingItem>
                <Item reward="'''+str(reward_waypoint)+'''" type="diamond_block"/>
              </RewardForCollectingItem>
              <RewardForSendingCommand reward="'''+str(reward_sendcommand)+'''"/>
              <RewardForMissionEnd rewardForDeath="-1000000">
                <Reward description="found_goal" reward="'''+str(reward_goal)+'''" />
                <Reward description="out_of_time" reward="'''+str(reward_timeout)+'''" />
              </RewardForMissionEnd>
              <AgentQuitFromTouchingBlockType>
                <Block type="redstone_block" description="found_goal" />
              </AgentQuitFromTouchingBlockType>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''
    return xml_str, msize.get(mission_type,100),reward_goal,reward_waypoint,n_intermediate_rewards,reward_timeout,reward_sendcommand,mtimeout.get(mission_type_tmp,0)

#----------------------------------------------------------------------------------------------------------------#
#-- This function initialises the mission based on the defined input arguments
def init_mission(agent_host, port=0, agent_type='Unknown',mission_type='Unknown', mission_seed=0, movement_type='Continuous'):
    """ Generate, and load the mission and return the agent host """

    #-- Sets up the mission via XML definition --#
    mission_xml, msize, reward_goal,reward_intermediate,n_intermediate_rewards,reward_timeout,reward_sendcommand, timeout = GetMissionInstance(mission_type,mission_seed,agent_type)
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
    my_mission.forceWorldReset()

    #-- Enforces the specific restrictions for the assessed exercise --#
    my_mission.setModeToCreative()
    if agent_type.lower()=='random':
        n = msize
        my_mission.observeGrid(-n, -1, -n, n, -1, n, 'grid')
        my_mission.requestVideoWithDepth(320,240)
    elif agent_type.lower()=='simple':
        n = msize
        my_mission.observeGrid(-n, -1, -n, n, -1, n, 'grid');
        my_mission.requestVideo(320,240)
    elif agent_type.lower()=='realistic':
        n = 1 # n=1 means local info only!
        my_mission.observeGrid(-n,-1,-n, n, -1, n, 'grid');
        my_mission.requestVideoWithDepth(320,240)
    elif agent_type.lower()=='helper':
        n = 100
        my_mission.observeGrid(-n,-1,-n, n, -1, n, 'grid');
        my_mission.requestVideoWithDepth(320,240)
    else:
        #-- Defines a custom agent and add the mecessary sensors --#
        n = 100
        my_mission.observeGrid(-n, -1, -n, n, 1, n, 'grid');
        my_mission.requestVideoWithDepth(320,240)

    #-- Add support for the specific movement type requested (and given the constraints of the assignment) --#
    #-- See e.g. http://microsoft.github.io/malmo/0.17.0/Schemas/MissionHandlers.html   --#
    if movement_type.lower()=='absolute':
        my_mission.allowAllAbsoluteMovementCommands()
    elif movement_type.lower()=='continuous':
        my_mission.allowContinuousMovementCommand('move')
        my_mission.allowContinuousMovementCommand('strafe')
        my_mission.allowContinuousMovementCommand('pitch')
        my_mission.allowContinuousMovementCommand('turn')
        my_mission.allowContinuousMovementCommand('crouch')
    elif movement_type.lower()=='discrete':
        my_mission.allowDiscreteMovementCommand('turn')
        my_mission.allowDiscreteMovementCommand('move')
        my_mission.allowDiscreteMovementCommand('movenorth')
        my_mission.allowDiscreteMovementCommand('moveeast')
        my_mission.allowDiscreteMovementCommand('movesouth')
        my_mission.allowDiscreteMovementCommand('movewest')
        my_mission.allowDiscreteMovementCommand('look')

    #-- Get the resulting xml (and return in order to check that conditions match the report) --#
    final_xml = my_mission.getAsXML(True)

    #-- Set up a recording if desired for later inspection
    my_mission_record = MalmoPython.MissionRecordSpec('tmp' + ".tgz")
    my_mission_record.recordRewards()
    my_mission_record.recordMP4(24,400000)

    #-- Attempt to start a mission --#
    max_retries = 5
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    #-- Loop until mission starts: --#
    print("Waiting for the mission to start ")
    state_t = agent_host.getWorldState()
    while not state_t.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        state_t = agent_host.getWorldState()
        for error in state_t.errors:
            print("Error:",error.text)

    print
    print( "Mission started (xml returned)... ")
    return final_xml,reward_goal,reward_intermediate,n_intermediate_rewards,reward_timeout,reward_sendcommand,timeout


#--------------------------------------------------------------------------------------
#-- This class implements the Realistic Agent --#
class AgentRealistic:

    def __init__(self,agent_host,agent_port, mission_type, mission_seed, solution_report, state_space_graph):
        """ Constructor for the realistic agent """
        self.AGENT_MOVEMENT_TYPE = 'Discrete' #This can be varied between the following -  {Absolute, Discrete, Continuous}
        self.AGENT_NAME = 'Realistic'
        self.AGENT_ALLOWED_ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

        self.agent_host = agent_host
        self.agent_port = agent_port
        self.mission_seed = mission_seed
        self.mission_type = mission_type
        self.state_space = None; # Note - To be a true Realistic Agent, it can not know anything about the state_space a priori!
        self.solution_report = solution_report;
        self.solution_report.setMissionType(self.mission_type)
        self.solution_report.setMissionSeed(self.mission_seed)
        self.last_reward = 0
        self.accumulative_reward = 0
        self.brain = Dqn(2, len(self.AGENT_ALLOWED_ACTIONS), 0.9)
        self.brain.load()


    #----------------------------------------------------------------------------------------------------------------#
    def __ExecuteActionForRealisticAgentWithNoisyTransitionModel__(self, idx_requested_action, noise_level):
        """ Creates a well-defined transition model with a certain noise level """
        n = len(self.AGENT_ALLOWED_ACTIONS)
        pp = noise_level/(n-1) * np.ones((n,1))
        pp[idx_requested_action] = 1.0 - noise_level
        idx_actual = np.random.choice(n, 1, p=pp.flatten()) # sample from the distribution of actions
        actual_action = self.AGENT_ALLOWED_ACTIONS[int(idx_actual)]
        self.agent_host.sendCommand(actual_action)
        return actual_action

    #----------------------------------------------------------------------------------------------------------------#
    def run_agent(self):
        """ Run the Realistic agent and log the performance and resource use """

        partialReward = 0
        #-- Load and initiate mission --#
        print('Generate and load the ' + self.mission_type + ' mission with seed ' + str(self.mission_seed) + ' allowing ' +  self.AGENT_MOVEMENT_TYPE + ' movements')
        mission_xml = init_mission(self.agent_host, self.agent_port, self.AGENT_NAME, self.mission_type, self.mission_seed, self.AGENT_MOVEMENT_TYPE)
        self.solution_report.setMissionXML(mission_xml)

        self.solution_report.start()
        time.sleep(1)

        state_t = self.agent_host.getWorldState()
        first = True
        # -- Get a state-space model by observing the Orcale/GridObserver--#
        while state_t.is_mission_running:
            if first:
                time.sleep(2)
                first = False
             # -- Basic map --#
            state_t = self.agent_host.getWorldState()  
            if state_t.number_of_observations_since_last_state > 0:
                msg = state_t.observations[-1].text  # Get the details for the last observed state
                oracle_and_internal = json.loads(msg)  # Parse the Oracle JSON
                grid = oracle_and_internal.get(u'grid', 1)
                xpos = oracle_and_internal.get(u'XPos', 1)
                zpos = oracle_and_internal.get(u'ZPos', 1)
                ypos = oracle_and_internal.get(u'YPos', 1)
                yaw = oracle_and_internal.get(u'Yaw', 1)  
                pitch = oracle_and_internal.get(u'Pitch', 1)

            #last_signal = [xpos, zpos, ypos, yaw, pitch]
            last_signal = [xpos, ypos]


            action = self.brain.update(self.last_reward, last_signal)
            print("Requested Action:", self.AGENT_ALLOWED_ACTIONS[action])
            self.__ExecuteActionForRealisticAgentWithNoisyTransitionModel__(action, 0.1)
            time.sleep(0.02)
            self.solution_report.action_count = self.solution_report.action_count + 1
            for reward_t in state_t.rewards:
                partialReward += reward_t.getValue()
                #self.last_reward = reward_t.getValue()

                self.accumulative_reward += reward_t.getValue()
                self.solution_report.addReward(reward_t.getValue(), datetime.datetime.now())
                print("Reward_t:",reward_t.getValue())
                print("Cummulative reward so far:", self.accumulative_reward)

            print("Last Reward:{0}".format(partialReward))
            self.last_reward = partialReward
            partialReward = 0


        return


#--------------------------------------------------------------------------------------
#-- This class implements the Simple Agent --#
class AgentSimple:

    def __init__(self,agent_host,agent_port, mission_type, mission_seed, solution_report, state_space):
        """ Constructor for the simple agent """
        self.AGENT_MOVEMENT_TYPE = 'Discrete' # Note - See comment @ Line 292
        self.AGENT_NAME = 'Simple'
        self.AGENT_ALLOWED_ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]


        self.agent_host = agent_host
        self.agent_port = agent_port
        self.mission_seed = mission_seed
        self.mission_type = mission_type
        self.state_space = state_space;
        self.solution_report = solution_report;
        self.solution_report.setMissionType(self.mission_type)
        self.solution_report.setMissionSeed(self.mission_seed)
    
    def __ExecuteActionForRandomAgentWithNoisyTransitionModel__(self, idx_request_action, noise_level):
        """ Creates a well-defined transition model with a certain noise level """
        n = len(self.AGENT_ALLOWED_ACTIONS)
        pp = noise_level/(n-1) * np.ones((n,1))
        pp[idx_request_action] = 1.0 - noise_level
        idx_actual = np.random.choice(n, 1, p=pp.flatten()) # sample from the distrbution of actions
        actual_action = self.AGENT_ALLOWED_ACTIONS[int(idx_actual)]
        self.agent_host.sendCommand(actual_action)
        return actual_action

    # Manhattan distance between goal and current state.
    def heuristic(self, node, goal, grid):
        curr_x = grid[node][0]
        curr_y = grid[node][0]
        goal_x = goal[0]
        goal_y = goal[1]
        return abs(curr_x - goal_x) + abs(curr_y - goal_y)
    
    # Translate two nodes into an action value.
    def direction(self, current, chosen):
        if chosen[0] > current[0]:
            return 3
        if chosen[0] < current[0]:
            return 2
        if chosen[1] > current[1]:
            return 1
        if chosen[1] < current[1]:
            return 0

    # Perform A* search and return path.
    def a_star_search(self, grid, transitions, goal, goalCoord, start):

        explored = []
        path_cost = 0
        
        came_from = {}
        current_node_cost={}
        
        frontier = []
        frontier.append(start)
        
        for node in grid:
            current_node_cost[node] = 100000
        
        current_node_cost[start] = 0

        while (len(frontier) > 0):
            
            frontier = sorted(frontier, key=lambda node: current_node_cost[node] + self.heuristic(node, goalCoord, grid))
            frontier.reverse()

            current = frontier.pop()
            
            
            if (current == goal):
                break

            explored.append(current)

            for node in transitions[current]:
                
                if (node in explored):
                    continue
                
                if (node not in frontier):
                    frontier.append(node)
                
                node_path_cost = current_node_cost[current] + 1
                if node_path_cost >= current_node_cost[node]:
                    continue

                came_from[node] = current
                current_node_cost[node] = node_path_cost

        path = [current]
        while(current in came_from):
            current = came_from[current]
            path.append(current)
        
        return path

    def run_agent(self):
        """ Run the Simple agent and log the performance and resource use """

        #-- Load and initiate mission --#
        print('Generate and load the ' + self.mission_type + ' mission with seed ' + str(self.mission_seed) + ' allowing ' +  self.AGENT_MOVEMENT_TYPE + ' movements')
        mission_xml = init_mission(self.agent_host, self.agent_port, self.AGENT_NAME, self.mission_type, self.mission_seed, self.AGENT_MOVEMENT_TYPE)
        self.solution_report.setMissionXML(mission_xml)
        time.sleep(1)
        self.solution_report.start()
        

        grid = deepcopy(self.state_space.state_locations)
        transitions = deepcopy(self.state_space.state_actions)
        
        initialState = deepcopy(self.state_space.start_id)
        initialCoord = deepcopy(self.state_space.start_loc)
        goalState = deepcopy(self.state_space.goal_id)
        goalCoord = deepcopy(self.state_space.goal_loc)
        
        reward_cumulative = 0
        
        state_t = self.agent_host.getWorldState()
        
        path = self.a_star_search(grid, transitions, goalState, goalCoord, initialState)
        print(path)

        current = grid[path.pop()]
        print(current)
        
        while state_t.is_mission_running and len(path) > 0:
            
            time.sleep(0.5)
            
            # Get the world state
            state_t = self.agent_host.getWorldState()
            
            new = grid[path.pop()]
            
            move = self.direction(current, new)
            
            print("Requested Action:",self.AGENT_ALLOWED_ACTIONS[move])
            actual_action = self.__ExecuteActionForRandomAgentWithNoisyTransitionModel__(move, 0);
            
            current = new

            for reward_t in state_t.rewards:
                reward_cumulative += reward_t.getValue()
                self.solution_report.addReward(reward_t.getValue(), datetime.datetime.now())
                print("Reward_t:",reward_t.getValue())
                print("Cummulative reward so far:",reward_cumulative)
            
            # Check if anything went wrong along the way
            for error in state_t.errors:
                print("Error:",error.text)
    
            # Handling of the sensor input
            xpos = None
            ypos = None
            zpos = None
            yaw  = None
            pitch = None
            if state_t.number_of_observations_since_last_state > 0: # Has any Oracle-like and/or internal sensor observations come in?
                msg = state_t.observations[-1].text      # Get the detailed for the last observed state
                oracle = json.loads(msg)                 # Parse the Oracle JSON
                
                # GPS-like sensor
                xpos = oracle.get(u'XPos', 0)            # Position in 2D plane, 1st axis
                zpos = oracle.get(u'ZPos', 0)            # Position in 2D plane, 2nd axis (yes Z!)
                ypos = oracle.get(u'YPos', 0)            # Height as measured from surface! (yes Y!)
                
                # Standard "internal" sensory inputs
                yaw  = oracle.get(u'Yaw', 0)             # Yaw
                pitch = oracle.get(u'Pitch', 0)          # Pitch
            
            # Vision
            if state_t.number_of_video_frames_since_last_state > 0: # Have any Vision percepts been registred ?
                frame = state_t.video_frames[0]
        
            #-- Print some of the state information --#
            print("Percept: video,observations,rewards received:",state_t.number_of_video_frames_since_last_state,state_t.  number_of_observations_since_last_state,state_t.number_of_rewards_since_last_state)
            print("\tcoordinates (x,y,z,yaw,pitch):" + str(xpos) + " " + str(ypos) + " " + str(zpos)+ " " + str(yaw) + " " + str(pitch))

        # --------------------------------------------------------------------------------------------
        # Summary
        print("Summary:")
        print("Cumulative reward = " + str(reward_cumulative) )
        
        return

#--------------------------------------------------------------------------------------
#-- This class implements a basic, suboptimal Random Agent. The purpurpose is to provide a baseline for other agent to beat. --#
class AgentRandom:

    def __init__(self,agent_host,agent_port, mission_type, mission_seed, solution_report, state_space_graph):
        """ Constructor for the Random agent """
        self.AGENT_MOVEMENT_TYPE = 'Discrete'
        self.AGENT_NAME = 'Random'
        self.AGENT_ALLOWED_ACTIONS = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]

        self.agent_host = agent_host
        self.agent_port = agent_port
        self.mission_seed = mission_seed
        self.mission_type = mission_type
        self.state_space = StateSpace()
        #self.state_space = state_space; ????????????????????????????????????????????????
        self.solution_report = solution_report;   # Python makes call by reference !
        self.solution_report.setMissionType(self.mission_type)
        self.solution_report.setMissionSeed(self.mission_seed)

    def __ExecuteActionForRandomAgentWithNoisyTransitionModel__(self, idx_request_action, noise_level):
        """ Creates a well-defined transition model with a certain noise level """
        n = len(self.AGENT_ALLOWED_ACTIONS)
        pp = noise_level/(n-1) * np.ones((n,1))
        pp[idx_request_action] = 1.0 - noise_level
        idx_actual = np.random.choice(n, 1, p=pp.flatten()) # sample from the distrbution of actions
        actual_action = self.AGENT_ALLOWED_ACTIONS[int(idx_actual)]
        self.agent_host.sendCommand(actual_action)
        return actual_action

    def run_agent(self):
        """ Run the Random agent and log the performance and resource use """

        #-- Load and init mission --#
        print('Generate and load the ' + self.mission_type + ' mission with seed ' + str(self.mission_seed) + ' allowing ' +  self.AGENT_MOVEMENT_TYPE + ' movements')
        mission_xml,reward_goal,reward_intermediate,n_intermediate_rewards,reward_timeout,reward_sendcommand, timeout = init_mission(self.agent_host, self.agent_port, self.AGENT_NAME, self.mission_type, self.mission_seed, self.AGENT_MOVEMENT_TYPE)
        self.solution_report.setMissionXML(mission_xml)

        #-- Define local capabilities of the agent (sensors)--#
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
        self.agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS)

        # Fix the randomness of the agent by seeding the random number generator
        reward_cumulative = 0.0

        # Main loop:
        state_t = self.agent_host.getWorldState()

        while state_t.is_mission_running:
            # Wait 0.5 sec
            time.sleep(0.5)

            # Get the world state
            state_t = self.agent_host.getWorldState()

            if state_t.is_mission_running:
                actionIdx = random.randint(0, 3)
                print("Requested Action:",self.AGENT_ALLOWED_ACTIONS[actionIdx])

                # Now try to execute the action given a noisy transition model
                actual_action = self.__ExecuteActionForRandomAgentWithNoisyTransitionModel__(actionIdx, 0);
                print("Actual Action:",actual_action)

            # Collect the number of rewards and add to reward_cumulative
            # Note: Since we only observe the sensors and environment every a number of rewards may have accumulated in the buffer
            for reward_t in state_t.rewards:
                reward_cumulative += reward_t.getValue()
                self.solution_report.addReward(reward_t.getValue(), datetime.datetime.now())
                print("Reward_t:",reward_t.getValue())
                print("Cummulative reward so far:",reward_cumulative)

            # Check if anything went wrong along the way
            for error in state_t.errors:
                print("Error:",error.text)

            # Handling of the sensor input
            xpos = None
            ypos = None
            zpos = None
            yaw  = None
            pitch = None
            if state_t.number_of_observations_since_last_state > 0: # Has any Oracle-like and/or internal sensor observations come in?
                msg = state_t.observations[-1].text      # Get the detailed for the last observed state
                oracle = json.loads(msg)                 # Parse the Oracle JSON

                # Orcale
                grid = oracle.get(u'grid', 0)            #

                # GPS-like sensor
                xpos = oracle.get(u'XPos', 0)            # Position in 2D plane, 1st axis
                zpos = oracle.get(u'ZPos', 0)            # Position in 2D plane, 2nd axis (yes Z!)
                ypos = oracle.get(u'YPos', 0)            # Height as measured from surface! (yes Y!)

                # Standard "internal" sensory inputs
                yaw  = oracle.get(u'Yaw', 0)             # Yaw
                pitch = oracle.get(u'Pitch', 0)          # Pitch

            # Vision
            if state_t.number_of_video_frames_since_last_state > 0: # Have any Vision percepts been registred ?
                frame = state_t.video_frames[0]

            #-- Print some of the state information --#
            print("Percept: video,observations,rewards received:",state_t.number_of_video_frames_since_last_state,state_t.number_of_observations_since_last_state,state_t.number_of_rewards_since_last_state)
            print("\tcoordinates (x,y,z,yaw,pitch):" + str(xpos) + " " + str(ypos) + " " + str(zpos)+ " " + str(yaw) + " " + str(pitch))

        # --------------------------------------------------------------------------------------------
        # Summary
        print("Summary:")
        print("Cumulative reward = " + str(reward_cumulative) )

        return


#--------------------------------------------------------------------------------------
#-- This class implements a helper Agent for deriving the state-space representation ---#
class AgentHelper:
    """ This agent determines the state space for use by the actual problem solving agent. Enabeling do_plot will allow you to visualize the results """

    def __init__(self,agent_host,agent_port, mission_type, mission_seed, solution_report, state_space_graph):
        """ Constructor for the helper agent """
        self.AGENT_NAME = 'Helper'
        self.AGENT_MOVEMENT_TYPE = 'Absolute' # Note the helper needs absolute movements
        self.DO_PLOT = False #Turn this to 'True' to generate a plot of the grid

        self.agent_host = agent_host
        self.agent_port = agent_port
        self.mission_seed = mission_seed
        self.mission_type = mission_type
        self.state_space = StateSpace()
        self.solution_report = solution_report;  
        self.solution_report.setMissionType(self.mission_type)
        self.solution_report.setMissionSeed(self.mission_seed)

    def run_agent(self):
        """ Run the Helper agent to get the state-space """

        #-- Load and init the Helper mission --#
        print('Generate and load the ' + self.mission_type + ' mission with seed ' + str(self.mission_seed) + ' allowing ' +  self.AGENT_MOVEMENT_TYPE + ' movements')
        mission_xml,reward_goal,reward_intermediate,n_intermediate_rewards,reward_timeout,reward_sendcommand, timeout = init_mission(self.agent_host, self.agent_port, self.AGENT_NAME, self.mission_type, self.mission_seed, self.AGENT_MOVEMENT_TYPE)
        self.solution_report.setMissionXML(mission_xml)

        #-- Define local capabilities of the agent (sensors)--#
        self.agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
        self.agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
        self.agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS)

        time.sleep(1)

        #-- Get the state of the world along with internal agent state...--#
        state_t = self.agent_host.getWorldState()
        print("here is the state of the agent:\n {0}".format(state_t))

        #-- Get a state-space model by observing the Orcale/GridObserver--#
        if state_t.is_mission_running:
            #-- Make sure we look in the right direction when observing the surrounding (otherwise the coordinate system will rotated by the Yaw !) --#
            # Look East (towards +x (east) and +z (south) on the right, i.e. a std x,y coordinate system) yaw=-90
            self.agent_host.sendCommand("setPitch 20")
            time.sleep(1)
            self.agent_host.sendCommand("setYaw -90")
            time.sleep(1)

            #-- Basic map --#
            state_t = self.agent_host.getWorldState()           

            if state_t.number_of_observations_since_last_state > 0:
                msg = state_t.observations[-1].text                 # Get the details for the last observed state
                oracle_and_internal = json.loads(msg)               # Parse the Oracle JSON
                grid = oracle_and_internal.get(u'grid', 1)          # This line is getting the grid from Oracle. But it is a list of "stone"????????????
                xpos = oracle_and_internal.get(u'XPos', 0)
                zpos = oracle_and_internal.get(u'ZPos', 0)
                ypos = oracle_and_internal.get(u'YPos', 0)
                yaw  = oracle_and_internal.get(u'Yaw', 0)      
                pitch = oracle_and_internal.get(u'Pitch', 0)        

                #-- Parste the JOSN string, Note there are better ways of doing this! --#
                full_state_map_raw = str(grid)
                full_state_map_raw=full_state_map_raw.replace("[","")
                full_state_map_raw=full_state_map_raw.replace("]","")
                full_state_map_raw=full_state_map_raw.replace("u'","")
                full_state_map_raw=full_state_map_raw.replace("'","")
                full_state_map_raw=full_state_map_raw.replace(" ","")
                aa=full_state_map_raw.split(",")
                vocs = list(set(aa))
                for word in vocs:
                    for i in range(0,len(aa)):
                        if aa[i]==word :
                            aa[i] = vocs.index(word)

                X = np.asarray(aa);
                nn = int(math.sqrt(X.size))
                X = np.reshape(X, [nn,nn]) # Note: this matrix/table is index as z,x

                #-- Visualize the discrete state-space --#
                if self.DO_PLOT:
                    print(yaw)
                    plt.figure(1)
                    imgplot = plt.imshow(X.astype('float'),interpolation='none')
                    plt.show()

                #-- Define the unique states available --#
                state_wall = vocs.index("stained_hardened_clay")
                state_impossible = vocs.index("stone")
                state_initial = vocs.index("emerald_block")
                state_goal = vocs.index("redstone_block")

                #-- Extract state-space --#
                offset_x = 100-math.floor(xpos);
                offset_z = 100-math.floor(zpos);

                state_space_locations = {}; # create a dict

                for i_z in range(0,len(X)):
                    for j_x in range(0,len(X)):
                        if X[i_z,j_x] != state_impossible and X[i_z,j_x] != state_wall:
                            state_id = "S_"+str(int(j_x - offset_x))+"_"+str(int(i_z - offset_z) )
                            state_space_locations[state_id] = (int(j_x- offset_x),int(i_z - offset_z) )
                            if X[i_z,j_x] == state_initial:
                                state_initial_id = state_id
                                loc_start = state_space_locations[state_id]
                            elif X[i_z,j_x] == state_goal:
                                state_goal_id = state_id
                                loc_goal = state_space_locations[state_id]

                #-- Generate state / action list --#
                # First define the set of actions in the defined coordinate system
                actions = {"west": [-1,0],"east": [+1,0],"north": [0,-1], "south": [0,+1]}
                state_space_actions = {}
                for state_id in state_space_locations:
                    possible_states = {}
                    for action in actions:
                        #-- Check if a specific action is possible --#
                        delta = actions.get(action)
                        state_loc = state_space_locations.get(state_id)
                        state_loc_post_action = [state_loc[0]+delta[0],state_loc[1]+delta[1]]

                        #-- Check if the new possible state is in the state_space, i.e., is accessible --#
                        state_id_post_action = "S_"+str(state_loc_post_action[0])+"_"+str(state_loc_post_action[1])
                        if state_space_locations.get(state_id_post_action) != None:
                            possible_states[state_id_post_action] = 1

                    #-- Add the possible actions for this state to the global dict --#
                    state_space_actions[state_id] = possible_states

                #-- Kill the agent/mission --#
                agent_host.sendCommand("tp " + str(0 ) + " " + str(0) + " " + str(0))
                time.sleep(2)

                #-- Save the info an instance of the StateSpace class --
                self.state_space.state_actions = state_space_actions
                self.state_space.state_locations = state_space_locations
                self.state_space.start_id = state_initial_id
                self.state_space.start_loc = loc_start
                self.state_space.goal_id  = state_goal_id
                self.state_space.goal_loc = loc_goal
        
        
    
            state_space_rewards = {}
            state_space_rewards[state_goal_id] = reward_goal

            #-- Set the values in the state_space container --#
            self.state_space.reward_states = state_space_rewards
            self.state_space.reward_states_n = n_intermediate_rewards + 1
            self.state_space.reward_timeout = reward_timeout
            self.state_space.timeout = timeout
            self.state_space.reward_sendcommand = reward_sendcommand
        else:
            self.state_space = None
            #-- End if observations --#

        return

# --------------------------------------------------------------------------------------------
#-- The main entry point if you run the module as a script--#
if __name__ == "__main__":

    #-- Define default arguments, in case you run the module as a script --#
    DEFAULT_STUDENT_GUID = 'template'
    DEFAULT_AGENT_NAME   = 'Random' #HINT: Currently choose between {Random,Simple, Realistic}
    DEFAULT_MALMO_PATH   = '/Users/USMAC/Desktop/Glasgow/AI/Malmo-0.17.0-Mac-64bit' # HINT: Change this to your own path
    DEFAULT_AIMA_PATH    = '/Users/USMAC/Desktop/Glasgow/AI/Labs/lab_3_search/aima-python/'  # HINT: Change this to your own path, forward slash only, should be the 2.7 version from https://www.dropbox.com/s/vulnv2pkbv8q92u/aima-python_python_v27_r001.zip?dl=0) or for Python 3.x get it from https://github.com/aimacode/aima-python
    DEFAULT_MISSION_TYPE = 'small'  #HINT: Choose between {small,medium,large}
    DEFAULT_MISSION_SEED_MAX = 1    #HINT: How many different instances of the given mission (i.e. maze layout)
    DEFAULT_REPEATS      = 1        #HINT: How many repetitions of the same maze layout
    DEFAULT_PORT         = 0
    DEFAULT_SAVE_PATH    = './results/'

    #-- Import required modules --#
    import os
    import sys
    import time
    import random
    from random import shuffle
    import json
    import argparse
    import pickle
    import datetime
    import math
    import numpy as np
    from copy import deepcopy
    import hashlib

    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib import lines
    from ai import Dqn

    #-- Define the commandline arguments required to run the agents from command line --#
    parser = argparse.ArgumentParser()
    parser.add_argument("-a" , "--agentname"        , type=str, help="path for the malmo pyhton examples"   , default=DEFAULT_AGENT_NAME)
    parser.add_argument("-t" , "--missiontype"      , type=str, help="mission type (small,medium,large)"    , default=DEFAULT_MISSION_TYPE)
    parser.add_argument("-s" , "--missionseedmax"   , type=int, help="maximum mission seed value (integer)"           , default=DEFAULT_MISSION_SEED_MAX)
    parser.add_argument("-n" , "--nrepeats"         , type=int, help="repeat of a specific agent (if stochastic behavior)"  , default=DEFAULT_REPEATS)
    parser.add_argument("-g" , "--studentguid"      , type=str, help="student guid"                         , default=DEFAULT_STUDENT_GUID)
    parser.add_argument("-p" , "--malmopath"        , type=str, help="path for the malmo pyhton examples"   , default=DEFAULT_MALMO_PATH)
    parser.add_argument("-x" , "--malmoport"        , type=int, help="special port for the Minecraft client", default=DEFAULT_PORT)
    parser.add_argument("-o" , "--aimapath"         , type=str, help="path for the aima toolbox (optional)"   , default=DEFAULT_AIMA_PATH)
    parser.add_argument("-r" , "--resultpath"       , type=str, help="the path where the results are saved" , default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()
    print("\nargs{0}\n".format(args))

    #-- Display infor about the system --#
    print("Working dir:"+os.getcwd())
    print("Python version:"+sys.version)
    print("malmopath:"+args.malmopath)
    #print("JAVA_HOME:'"+os.environ["JAVA_HOME"]+"'")
    #print("MALMO_XSD_PATH:'"+os.environ["MALMO_XSD_PATH"]+"'")

    #-- Add the Malmo path  --#
    print('Add Malmo Python API/lib to the Python environment ['+args.malmopath+'/Python_Examples'+']')
    sys.path.append(args.malmopath+'Python_Examples/')

    #-- Import the Malmo Python wrapper/module --#
    print('Import the Malmo module...')
    import MalmoPython

    #-- OPTIONAL: Import the AIMA tools (for representing the state-space)--#
    #print('Add AIMA lib to the Python environment ['+args.aimapath+']')
    #sys.path.append(args.aimapath+'/')
    #from search import *

    #-- Create the command line string for convenience --#
    cmd = 'python myagents.py -a ' + args.agentname + ' -s ' + str(args.missionseedmax) + ' -n ' + str(args.nrepeats) + ' -t ' + args.missiontype + ' -g ' + args.studentguid + ' -p ' + args.malmopath + ' -x ' + str(args.malmoport)
    print(cmd)

    #-- Run the agent a number of times (it only makes a difference if you agent has some random elemnt to it, initalizaiton, behavior, etc.) --#
    #-- HINT: It is quite important that you understand the need for the loops  --#
    #-- HINT: Depending on how you implement your realistic agent in terms of restarts and repeats, you may want to change the way the loops operate --#

    print('Instantiate an agent interface/api to Malmo')
    agent_host = MalmoPython.AgentHost()

    #-- Itereate a few different layout of the same mission stype --#
    for i_training_seed in range(0,args.missionseedmax):

        #-- Observe the full state space a prior i (only allowed for the simple agent!) ? --#
        if args.agentname.lower()=='helper' or args.agentname.lower()=='simple':
            print('Get state-space representation using a AgentHelper...[note in v0.30 there is now an faster way of getting the state-space ]')
            helper_solution_report = SolutionReport()
            helper_agent = AgentHelper(agent_host,args.malmoport,args.missiontype,i_training_seed, helper_solution_report, None)
            helper_agent.run_agent()
        else:
            helper_agent = None
        """else:
            if args.agentname.lower()=='realistic':
                print('Get state-space representation using a AgentRealistic...[note in v0.30 there is now an faster way of getting the state-space ]')
                helper_solution_report = SolutionReport()
                helper_agent = AgentRealistic(agent_host, args.malmoport, args.missiontype, i_training_seed, helper_solution_report, None)

            elif args.agentname.lower()=='random':
                print(
                'Get state-space representation using a AgentRandom...[note in v0.30 there is now an faster way of getting the state-space ]')
                helper_solution_report = SolutionReport()
                helper_agent = AgentRandom(agent_host, args.malmoport, args.missiontype, i_training_seed, helper_solution_report, None)"""


        #-- Repeat the same instance (size and seed) multiple times --#
        run_instances = 0
        
        cumulativeRewards = []
        timings = []
        

        for i_rep in range(0,args.nrepeats):
            print('Setup the performance log...')
            solution_report = SolutionReport()
            solution_report.setStudentGuid(args.studentguid)

            print('Get an instance of the specific ' + args.agentname + ' agent with the agent_host and load the ' + args.missiontype + ' mission with seed ' + str(i_training_seed))
            agent_name = 'Agent' + args.agentname
            state_space = None;
            if not helper_agent==None:
                state_space = deepcopy(helper_agent.state_space)

            agent_to_be_evaluated = eval(agent_name+'(agent_host,args.malmoport,args.missiontype,i_training_seed,solution_report,state_space)')

            print('Run the agent, time it and log the performance...')
            solution_report.start() # start the timer (may be overwritten in the agent to provide a fair comparison)
            agent_to_be_evaluated.run_agent()

            solution_report.stop() # stop the timer

            run_instances += 1

            print("\n---------------------------------------------")
            print("| Solution Report Summary: ")
            print("|\tNumber of Instances Run = ") + str(run_instances)
            print("|\tCumulative reward = " + str(solution_report.reward_cumulative))
            print("|\tDuration (wallclock) = " + str((solution_report.end_datetime_wallclock-solution_report.start_datetime_wallclock).total_seconds()))
            print("|\tNumber of reported actions = " + str(solution_report.action_count))
            print("|\tFinal goal reached = " + str(solution_report.is_goal))
            print("|\tTimeout = " + str(solution_report.is_timeout))
            print("---------------------------------------------\n")


            # Outputs raw result data in CSV form to text file for each instance for analsysis and graphing

            cumulativeRewards.append(solution_report.reward_cumulative)

            file = open(str(agent_name) + " TestResults.txt", "a")
            
            # Outputs counter for instance of agent in loop, time taken, reward, and no of actions required.
            file.write(str(run_instances) + "," + str(((solution_report.end_datetime_wallclock-solution_report.start_datetime_wallclock).total_seconds())) + "," + str(solution_report.reward_cumulative) + "," + str(solution_report.action_count) + "\n")
            file.close()
            print('Instance report saved for later analysis and reporting')

            print('Sleep a sec to make sure the client is ready for next mission/agent variation...')
            time.sleep(1)
            print("------------------------------------------------------------------------------\n")
            if isinstance(agent_to_be_evaluated, AgentRealistic):
                print("score form brain:{0}".format(agent_to_be_evaluated.brain.score()))
                agent_to_be_evaluated.brain.save()

        plt.plot(cumulativeRewards)
        plt.ylabel('cumulative rewards')
        plt.savefig("{0}.pdf".format(args.nrepeats))
        plt.show()





    print("Done")
