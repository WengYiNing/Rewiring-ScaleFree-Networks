#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <limits>
#include <fstream>
#include <set>
#include <list>
#include <filesystem>

using namespace std;

static std::mt19937 rng;

void sgenrand(unsigned int seed) {
    rng.seed(seed);
}

double randf() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

long randi(long LIM) {
    std::uniform_int_distribution<long> dist(0, LIM - 1);
    return dist(rng);
}
//------------------------------------------------------------------

const int SEED = 10;

// PDG payoff parameters
double Dr = 0.14;
double Dg = Dr;
const double gamma = 0.9;
const double epsilon = 0.02;
const double alpha = 0.1; // learning rate

// Prisoner's Dilemma payoffs
const double R = 1.0;     // mutual cooperation reward
const double P = 0.0;     // mutual defection punishment
const double S = -Dr;     // sucker's payoff
const double T = 1 + Dg;  // temptation to defect

using namespace std;
using State = int;


enum Action { COOPERATE, DEFECT };
enum RewireState {REWIRE, NOTREWIRE};
enum AgentType{ NORMAL, UNIQUE };

// Simulation settings
const int SIZE = 2500;
const int totalRound = 10000000;
const int focusRound = 5000;

// Initial number of nodes in seed network
const int m0 = 5;

// Number of neighbors per new node (BA model parameter m)
const int m = 2;

// Global total degree counter
int totalDegree = 0;

// Global agent index counter
int agentCnt = 0;

// Rewiring constraints
const int CONSTRAIN = 1;

// rewiring cost
const double COST = 0.4;

// Round counters
int neighborRound = 0;

int rewireCnt = 0;
int gotRewiredCnt = 0;

// Agent definition
class Agent
{
public:
    AgentType agentType;
    list<Agent*> neighbors;
    int index;
    

    double uniqueQtable[2][3];
    list<State> uniqueCurrentStates;
    list<State> uniqueNewStates;
    list<double> uniqueRewards;
    list<double> periodRewards;
    
    // Rewire decision Q-table and flags
    double rewireQtable[2][3];
    list<RewireState> ifRewire;
    
    // Actions to neighbors (last and new)
    list<Action> lastActions;
    list<Action> newActions;
    
    // Neighbors' actions received (pointers to their action containers)
    list<Action*> newActionRecieveList;
    list<Action*> lastActionRecieveList;

    int degree = 0;

    Agent();
    ~Agent();
    void setAgentType(AgentType A);
    AgentType getAgentType();
    
    void setNeighbor(Agent &a);
    void putNeighborRounds(int rounds);
    
    Action getNewAction(int index) const;
    Action getLastAction(int index) const;

    void setNewAction(Action a);
    void setLastAction(Action a);
    
    void putNewAction(Action a, int index);
    void putLastAction();
    
    bool isNeighbor(Agent* a);

    void computeReward_unique();
    
    void setCurrentState_unique();
    State getCurrentState_unique(int index) const;
    void setNewState_unique(State s, int d);
    State getNewState_unique(int index) const;
    int computeState_unique(int index);
    
    void chooseAction_unique();
    
    void updateQValue_unique();
    
    void updateRewireQValue();
    void chooseIfRewire();
    void rewire(const list<Agent*>& network, const list<Agent*>& tempList);
    void gotRewired();
};

// Agent methods
Agent::Agent()
{
    agentType = NORMAL;
    index = 0;

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            uniqueQtable[i][j] = 0;
        }
    }
    
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            rewireQtable[i][j] = 0;
        }
    }
}

Agent::~Agent()
{
}

void Agent::setAgentType(AgentType A)
{
    agentType = A;
}

AgentType Agent::getAgentType()
{
    return agentType;
}

void Agent::setNeighbor(Agent &a)
{
    neighbors.push_back(&a);
}

Action Agent::getNewAction(int index) const
{
    return *next(newActions.begin(), index);
}

Action Agent::getLastAction(int index) const
{
    return *next(lastActions.begin(), index);
}

void Agent::setNewAction(Action a)
{
    newActions.push_back(a);
}

void Agent::setLastAction(Action a)
{
    lastActions.push_back(a);
}

void Agent::putNewAction(Action a, int index)
{
    *next(newActions.begin(), index) = a;
}

void Agent::putLastAction()
{
    auto itLastAction = lastActions.begin();
    auto itNewAction = newActions.begin();
    
    for(int i = 0; i < degree; i++)
    {
        *itLastAction = *itNewAction;
        
        ++itLastAction;
        ++itNewAction;
    }
}

bool Agent::isNeighbor(Agent* a)
{
    return find(neighbors.begin(), neighbors.end(), a) != neighbors.end();
}

// Compute per-edge rewards based on own action and neighbor's action
void Agent::computeReward_unique()
{
    double reward = 0;
    auto itAction = newActions.begin();
    auto itNeighbor = newActionRecieveList.begin();
    auto itRewards = uniqueRewards.begin();
    auto itPeriodRewards = periodRewards.begin();
    
    for(int i = 0; i < degree; i++)
    {
        reward = 0;
        
        if(*itAction == COOPERATE && **itNeighbor == COOPERATE)
        {
            reward += R;
        }
        else if(*itAction == COOPERATE && **itNeighbor == DEFECT)
        {
            reward += S;
        }
        else if(*itAction == DEFECT && **itNeighbor == COOPERATE)
        {
            reward += T;
        }
        else if(*itAction == DEFECT && **itNeighbor == DEFECT)
        {
            reward += P;
        }
        
        *itRewards = reward;
        *itPeriodRewards += reward;
        
        ++itAction;
        ++itNeighbor;
        ++itRewards;
        ++itPeriodRewards;
    }
}

// Copy new states to current states
void Agent::setCurrentState_unique()
{
    auto itCurrentStares = uniqueCurrentStates.begin();
    auto itNewStates = uniqueNewStates.begin();
    
    for(int i = 0; i < degree; i++)
    {
        *itCurrentStares = *itNewStates;
        
        ++itCurrentStares;
        ++itNewStates;
    }
}

State Agent::getCurrentState_unique(int index) const
{
    auto it = uniqueCurrentStates.begin();
    std::advance(it, index);
    return *it;
}


void Agent::setNewState_unique(State s, int d)
{
    auto it = uniqueNewStates.begin();
    std::advance(it, d);
    *it = s;
}

State Agent::getNewState_unique(int index) const
{
    auto it = uniqueNewStates.begin();
    std::advance(it, index);
    return *it;
}

int Agent::computeState_unique(int index)
{
 
    int cooperatingNeighbors = 0;
    
    auto itNeighborAction = next(newActionRecieveList.begin(), index);
    auto itAction = next(newActions.begin(), index);
    
    if(**itNeighborAction == COOPERATE)
        cooperatingNeighbors++;
    
    if(*itAction == COOPERATE)
        cooperatingNeighbors++;
    
    return cooperatingNeighbors;
}

void Agent::chooseAction_unique()
{
    auto itNewAction = newActions.begin();
    auto itStates = uniqueNewStates.begin();
    
    for(int i = 0; i < degree; i++)
    {
        if(randf() < epsilon)
        {
            *itNewAction = Action(randi(2));
        }
        else if(uniqueQtable[0][*itStates] == uniqueQtable[1][*itStates])
        {
            *itNewAction = Action(randi(2));
        }
        else
        {
            Action temp = COOPERATE;
            double maxQ = uniqueQtable[0][*itStates];
            
            for(int i = 0; i <= 1; i++)
            {
                if(uniqueQtable[i][*itStates] > maxQ)
                {
                    maxQ = uniqueQtable[i][*itStates];
                    temp = Action(i);
                }
            }
            
            *itNewAction = temp;
        }
        
        ++itNewAction;
        ++itStates;
    }
}

void Agent::updateQValue_unique()
{
    double transform[2][3];
    
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            transform[i][j] = 0;
        }
    }
    
    auto itNewStates = uniqueNewStates.begin();
    auto itCurrentStates = uniqueCurrentStates.begin();
    auto itRewards = uniqueRewards.begin();
    auto itActions = lastActions.begin();
    
    for(int j = 0; j < degree; j++)
    {
        double nextQ = uniqueQtable[0][*itNewStates];
        double variance = 0;
        
        for(int i = 0; i <= 1; i++)
        {
            if(uniqueQtable[i][*itNewStates] > nextQ)
            {
                nextQ = uniqueQtable[i][*itNewStates];
            }
        }
        
        variance = alpha*((*itRewards) + (gamma)*(nextQ) - uniqueQtable[*itActions][*itCurrentStates]);
        
        transform[*itActions][*itCurrentStates] += variance;
        
        ++itNewStates;
        ++itCurrentStates;
        ++itRewards;
        ++itActions;
    }

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            uniqueQtable[i][j] += transform[i][j];
        }
    }
}

void Agent::updateRewireQValue()
{
    double transform[2][3];
    
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            transform[i][j] = 0;
        }
    }
    
    auto itNewStates = uniqueNewStates.begin();
    auto itCurrentStates = uniqueCurrentStates.begin();
    auto itPeriodRewards = periodRewards.begin();
    auto itRewire = ifRewire.begin();


    for(int j = 0; j < degree; j++)
    {
        
        double nextQ = rewireQtable[0][*itNewStates];
        double variance = 0;
        
        for(int i = 0; i <= 1; i++)
        {
            if(rewireQtable[i][*itNewStates] > nextQ)
            {
                nextQ = rewireQtable[i][*itNewStates];
            }
        }
        
        // Apply rewiring cost if REWIRE was chosen
        if(*itRewire == REWIRE)
        {
            (*itPeriodRewards) -= COST * CONSTRAIN;
        }
        
        variance = alpha*((*itPeriodRewards) + (gamma)*(nextQ) - rewireQtable[*itRewire][*itCurrentStates]);
        transform[*itRewire][*itCurrentStates] += variance;
    
        if(*itRewire == REWIRE)
        {
            (*itRewire) = NOTREWIRE;
        }
        
        *itPeriodRewards = 0;
        
        ++itNewStates;
        ++itCurrentStates;
        ++itPeriodRewards;
        ++itRewire;
    }

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            rewireQtable[i][j] += transform[i][j];
        }
    }
}

void Agent::chooseIfRewire()
{
    auto itRewire = ifRewire.begin();
    auto itActionRecieved = newActionRecieveList.begin();
    auto itAction = newActions.begin();
    auto itNewStates = uniqueNewStates.begin();
    
    while (itRewire != ifRewire.end())
    {
        RewireState temp = REWIRE;
        
        if(*itAction == COOPERATE && **itActionRecieved == DEFECT)
        {
            // epsilon chance to choose randomly
            if(randf() < epsilon)
            {
                *itRewire = RewireState(randi(2));
            }
            else
            {
                // tie-break random choice if Q values equal
                if(rewireQtable[0][*itNewStates] == rewireQtable[1][*itNewStates])
                {
                    *itRewire = RewireState(randi(2));
                }
                else
                {
                    double maxQ = rewireQtable[0][*itNewStates];
                    
                    for(int i = 0; i <= 1; i++)
                    {
                        if(rewireQtable[i][*itNewStates] > maxQ)
                        {
                            maxQ = rewireQtable[i][*itNewStates];
                            temp = RewireState(i);
                        }
                    }
                    
                    *itRewire = temp;
                }
            }
        }
        else
        {
            *itRewire = NOTREWIRE;
        }
        ++itRewire;
        ++itNewStates;
        ++itAction;
        ++itActionRecieved;
    }
}

void Agent::rewire(const list<Agent*>& network, const list<Agent*>& tempList)
{
    int maxIterations = int(ifRewire.size());
    int iterationCount = 0;
    
    auto itRewire = ifRewire.begin();
    auto agentDeleted = neighbors.begin();
    auto rewardDeleted = periodRewards.begin();
    auto uniqueRewardDeleted = uniqueRewards.begin();
    auto newStatesDeleted = uniqueNewStates.begin();
    auto currentStatesDeleted = uniqueCurrentStates.begin();
    auto lastActionDeleted = lastActions.begin();
    auto newActionDeleted = newActions.begin();
    auto newActionReceiveDeleted = newActionRecieveList.begin();
    auto lastActionReceiveDeleted = lastActionRecieveList.begin();
    
    while (itRewire != ifRewire.end() && iterationCount < maxIterations)
    {
        RewireState tempStatess = *itRewire;
        
        if (*itRewire == REWIRE)
        {
            if(!((*agentDeleted)->isNeighbor(this)))
            {
                *itRewire = NOTREWIRE;
                ++itRewire;
                ++agentDeleted;
                ++rewardDeleted;
                ++uniqueRewardDeleted;
                ++newStatesDeleted;
                ++currentStatesDeleted;
                ++lastActionDeleted;
                ++newActionDeleted;
                ++newActionReceiveDeleted;
                ++lastActionReceiveDeleted;
                iterationCount++;
                continue;
            }
            
            // Randomly select a new neighbor from the network: not self, not in tempList, not already connected
            Agent* newNeighbor = *next(network.begin(), randi(SIZE));
            
            // Try limited times to find a valid candidate matching "majority action" heuristic
            int maxTries = 200;
            
            Action tempActionNeighbor = COOPERATE;
            int cooperateCntNeighbor = 0, defectCntNeighbor = 0;
            
            for(int i = 0; i < newNeighbor->degree; i++)
            {
                if(*next(newNeighbor->newActions.begin(), i) == COOPERATE)
                {
                    cooperateCntNeighbor++;
                }
                else
                {
                    defectCntNeighbor++;
                }
            }
            
            if(cooperateCntNeighbor < defectCntNeighbor)
            {
                tempActionNeighbor = DEFECT;
            }
            
            Action myAction = COOPERATE;
            int myCooperateCnt = 0;
            int myDefectCnt = 0;
            
            for(int i = 0; i < degree - 1; i++)
            {
                if(*next(newActions.begin(), i) == COOPERATE)
                {
                    myCooperateCnt++;
                }
                else
                {
                    myDefectCnt++;
                }
            }
            
            if(myCooperateCnt < myDefectCnt)
            {
                myAction = DEFECT;
            }
            
            while((newNeighbor == this || find(tempList.begin(), tempList.end(), newNeighbor) != tempList.end() || newNeighbor->isNeighbor(this) || tempActionNeighbor != myAction) && maxTries-- > 0)
            {
                newNeighbor = *next(network.begin(), randi(SIZE));
                
                tempActionNeighbor = COOPERATE;
                cooperateCntNeighbor = 0;
                defectCntNeighbor = 0;
                
                for(int i = 0; i < newNeighbor->degree; i++)
                {
                    if(*next(newNeighbor->newActions.begin(), i) == COOPERATE)
                    {
                        cooperateCntNeighbor++;
                    }
                    else
                    {
                        defectCntNeighbor++;
                    }
                }
                
                if(cooperateCntNeighbor < defectCntNeighbor)
                {
                    tempActionNeighbor = DEFECT;
                }
            }
            
            if(maxTries <= 0)
            {
                *itRewire = NOTREWIRE;
                ++itRewire;
                ++agentDeleted;
                ++uniqueRewardDeleted;
                ++rewardDeleted;
                ++newStatesDeleted;
                ++currentStatesDeleted;
                ++lastActionDeleted;
                ++newActionDeleted;
                ++newActionReceiveDeleted;
                ++lastActionReceiveDeleted;
                iterationCount++;
                continue;
            }
            
            // Remove the old neighbor (edge) and corresponding per-edge containers
            agentDeleted = neighbors.erase(agentDeleted);
            rewardDeleted = periodRewards.erase(rewardDeleted);
            uniqueRewardDeleted = uniqueRewards.erase(uniqueRewardDeleted);
            newStatesDeleted = uniqueNewStates.erase(newStatesDeleted);
            currentStatesDeleted = uniqueCurrentStates.erase(currentStatesDeleted);
            lastActionDeleted = lastActions.erase(lastActionDeleted);
            newActionDeleted = newActions.erase(newActionDeleted);
            newActionReceiveDeleted = newActionRecieveList.erase(newActionReceiveDeleted);
            lastActionReceiveDeleted = lastActionRecieveList.erase(lastActionReceiveDeleted);
            itRewire = ifRewire.erase(itRewire);
            
            rewireCnt++;
            
            // Connect to the new neighbor (both directions)
            this->setNeighbor(*newNeighbor);
            
            newNeighbor->setLastAction(tempActionNeighbor);
            newNeighbor->setNewAction(tempActionNeighbor);
            this->lastActionRecieveList.push_back(&*std::prev(newNeighbor->lastActions.end()));
            this->newActionRecieveList.push_back(&*std::prev(newNeighbor->newActions.end()));
            newNeighbor->ifRewire.push_back(NOTREWIRE);
            
            newNeighbor->setNeighbor(*this);
            
            this->setLastAction(myAction);
            this->setNewAction(myAction);
            newNeighbor->lastActionRecieveList.push_back(&*std::prev(this->lastActions.end()));
            newNeighbor->newActionRecieveList.push_back(&*std::prev(this->newActions.end()));
            this->ifRewire.push_back(REWIRE);
            newNeighbor->degree++;

            this->uniqueRewards.push_back(0);
            this->periodRewards.push_back(0);
            this->uniqueCurrentStates.push_back(0);
            this->uniqueNewStates.push_back(this->computeState_unique(degree - 1));

            newNeighbor->uniqueRewards.push_back(0);
            newNeighbor->periodRewards.push_back(0);
            newNeighbor->uniqueCurrentStates.push_back(0);
            newNeighbor->uniqueNewStates.push_back(newNeighbor->computeState_unique(newNeighbor->degree - 1));

            iterationCount++;
            continue;
        }

        ++itRewire;
        ++agentDeleted;
        ++rewardDeleted;
        ++uniqueRewardDeleted;
        ++newStatesDeleted;
        ++currentStatesDeleted;
        ++lastActionDeleted;
        ++newActionDeleted;
        ++newActionReceiveDeleted;
        ++lastActionReceiveDeleted;
        iterationCount++;
    }
}

void Agent::gotRewired()
{
    auto agentDeleted = neighbors.begin();
    auto itRewire = ifRewire.begin();
    auto uniqueRewardDeleted = uniqueRewards.begin();
    auto rewardDeleted = periodRewards.begin();
    auto newStatesDeleted = uniqueNewStates.begin();
    auto currentStatesDeleted = uniqueCurrentStates.begin();
    auto lastActionDeleted = lastActions.begin();
    auto newActionDeleted = newActions.begin();
    auto newActionReceiveDeleted = newActionRecieveList.begin();
    auto lastActionReceiveDeleted = lastActionRecieveList.begin();

    while (agentDeleted != neighbors.end())
    {
        if (!((*agentDeleted)->isNeighbor(this)))
        {
            gotRewiredCnt++;
            
            agentDeleted = neighbors.erase(agentDeleted);
            uniqueRewardDeleted = uniqueRewards.erase(uniqueRewardDeleted);
            rewardDeleted = periodRewards.erase(rewardDeleted);
            newStatesDeleted = uniqueNewStates.erase(newStatesDeleted);
            currentStatesDeleted = uniqueCurrentStates.erase(currentStatesDeleted);
            lastActionDeleted = lastActions.erase(lastActionDeleted);
            newActionDeleted = newActions.erase(newActionDeleted);
            newActionReceiveDeleted = newActionRecieveList.erase(newActionReceiveDeleted);
            lastActionReceiveDeleted = lastActionRecieveList.erase(lastActionReceiveDeleted);
            itRewire = ifRewire.erase(itRewire);
            degree--;
            continue;
        }

        ++itRewire;
        ++agentDeleted;
        ++uniqueRewardDeleted;
        ++rewardDeleted;
        ++newStatesDeleted;
        ++currentStatesDeleted;
        ++lastActionDeleted;
        ++newActionDeleted;
        ++newActionReceiveDeleted;
        ++lastActionReceiveDeleted;
        
    }
}

class PDG
{
private:
    list<Agent*> network;
    vector<int> preferentialList;
    vector<double> cooperationRates;
public:
    PDG();
    ~PDG();
    void runGame();
    void saveCooperationRatesToFile(const string& filename);
    void initialize();
};


PDG::PDG()
{
    for (size_t i = 0; i < m0; i++)
    {
        Agent* tempAgent = new Agent();
        network.push_back(tempAgent);
        
        tempAgent->index = agentCnt;
        agentCnt++;
        tempAgent->setAgentType(UNIQUE);
    }
    
    // Fully connect the initial m0 nodes
    for(auto it1 = network.begin(); it1 != network.end(); ++it1)
    {
        for(auto it2 = std::next(it1); it2 != network.end(); ++it2)
        {
            (*it1)->setNeighbor(**it2);
            (*it2)->setLastAction(Action(randi(2)));
            (*it2)->setNewAction(Action(randi(2)));
            (*it1)->lastActionRecieveList.push_back(&*std::prev((*it2)->lastActions.end()));
            (*it1)->newActionRecieveList.push_back(&*std::prev((*it2)->newActions.end()));
            (*it1)->ifRewire.push_back(NOTREWIRE);
            (*it2)->degree++;

            (*it1)->uniqueRewards.push_back(0);
            (*it1)->periodRewards.push_back(0);
            (*it1)->uniqueCurrentStates.push_back(0);
            (*it1)->uniqueNewStates.push_back(0);

            totalDegree++;
            preferentialList.push_back(static_cast<int>(distance(network.begin(), it1)));
            
            (*it2)->setNeighbor(**it1);
            (*it1)->setLastAction(Action(randi(2)));
            (*it1)->setNewAction(Action(randi(2)));
            (*it2)->lastActionRecieveList.push_back(&*std::prev((*it1)->lastActions.end()));
            (*it2)->newActionRecieveList.push_back(&*std::prev((*it1)->newActions.end()));
            (*it2)->ifRewire.push_back(NOTREWIRE);
            (*it1)->degree++;
            
            (*it2)->uniqueRewards.push_back(0);
            (*it2)->periodRewards.push_back(0);
            (*it2)->uniqueCurrentStates.push_back(0);
            (*it2)->uniqueNewStates.push_back(0);

            totalDegree++;
            preferentialList.push_back(static_cast<int>(distance(network.begin(), it2)));
        }
    }
    
    // Preferential attachment growth to SIZE nodes
    for (int i = m0; i < SIZE; i++)
    {
        Agent* tempAgent = new Agent();
        network.push_back(tempAgent);
        
        tempAgent->index = agentCnt;
        agentCnt++;

        tempAgent->setAgentType(UNIQUE);
        
        std::vector<Agent*> chosenTargets;
        std::set<Agent*> chosenSet;

        while (chosenTargets.size() < m)
        {
            int randomNumber = int(randi(totalDegree));
            Agent* target = *std::next(network.begin(), preferentialList[randomNumber]);

            if (chosenSet.find(target) == chosenSet.end())
            {
                chosenTargets.push_back(target);
                chosenSet.insert(target);
            }
        }
        
        for(Agent* target : chosenTargets)
        {
            tempAgent->setNeighbor(*target);
            target->setLastAction(Action(randi(2)));
            target->setNewAction(Action(randi(2)));
            tempAgent->lastActionRecieveList.push_back(&*std::prev(target->lastActions.end()));
            tempAgent->newActionRecieveList.push_back(&*std::prev(target->newActions.end()));
            tempAgent->ifRewire.push_back(NOTREWIRE);
            tempAgent->degree++;

            tempAgent->uniqueRewards.push_back(0);
            tempAgent->periodRewards.push_back(0);
            tempAgent->uniqueCurrentStates.push_back(0);
            tempAgent->uniqueNewStates.push_back(0);
            
            totalDegree++;
            preferentialList.push_back(static_cast<int>(distance(network.begin(), std::prev(network.end()))));
            
            target->setNeighbor(*tempAgent);
            tempAgent->setLastAction(Action(randi(2)));
            tempAgent->setNewAction(Action(randi(2)));
            target->lastActionRecieveList.push_back(&*std::prev(tempAgent->lastActions.end()));
            target->newActionRecieveList.push_back(&*std::prev(tempAgent->newActions.end()));
            target->ifRewire.push_back(NOTREWIRE);
            target->degree++;

            target->uniqueRewards.push_back(0);
            target->periodRewards.push_back(0);
            target->uniqueCurrentStates.push_back(0);
            target->uniqueNewStates.push_back(0);
            
            totalDegree++;
            preferentialList.push_back(target->index);
        }
    }
}

PDG::~PDG()
{
    for (Agent* agent : network)
    {
        delete agent;
    }
    
    network.clear();
}

void PDG::saveCooperationRatesToFile(const string& filename)
{
    std::ofstream outFile(filename);
    
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    for (const auto& rate : cooperationRates)
    {
        outFile << rate << "\n";
    }
    outFile.close();
}

void PDG::runGame()
{
    std::filesystem::create_directories("results");
    
    std::string fileName = std::string("results/")
                           + "Dr="
                           + to_string(Dr)
                           + ", RC="
                           + to_string(CONSTRAIN)
                           + ", COST="
                           + to_string(COST)
                           + ", Nodes="
                           + to_string(SIZE)
                           + ", Seed="
                           + to_string(SEED)
                           + ", Rounds="
                           + to_string(totalRound)
                           + ".txt";

    std::ofstream outputFile(fileName);
    
    double cooperCnt = 0;
    
    for (int k = 0; k < totalRound - focusRound; ++k)
    {
        
        neighborRound++;
        double currentCooperationCount = 0;
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            for(int d = 0; d < agent->degree; d++)
            {
                agent->setNewState_unique(agent->computeState_unique(d), d);
            }
            agent->computeReward_unique();
            agent->putLastAction();
        }
        
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            agent->updateQValue_unique();
            
            if(neighborRound == CONSTRAIN && agent->degree < SIZE - 1)
            {
                agent->updateRewireQValue();
                agent->chooseIfRewire();
            }
        }
        
        if(neighborRound == CONSTRAIN)
        {
            for (auto it = network.begin(); it != network.end(); ++it)
            {
                Agent* agent = *it;
                
                if(agent->neighbors.empty())
                {
                    continue;
                }
                
                if(agent->degree < SIZE - 1)
                {
                    list<Agent*> tempList = agent->neighbors;
                    agent->rewire(network, tempList);
                }
                
            }
            
            for (auto it = network.begin(); it != network.end(); ++it)
            {
                Agent* agent = *it;
                if(agent->neighbors.empty())
                {
                    continue;
                }

                agent->gotRewired();
            }
            
            neighborRound = 0;
        }
        
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            agent->chooseAction_unique();
            agent->setCurrentState_unique();
            
            for(int d = 0; d < agent->degree; d++)
            {
                if(agent->getNewAction(d) == COOPERATE)
                    currentCooperationCount += 1;
            }
        }
        double currentCooperationRate = (currentCooperationCount) / (SIZE * 4);
        cooperationRates.push_back(currentCooperationRate);
    }
    
    for (int k = 0; k < focusRound; ++k)
    {
        neighborRound++;
        double currentCooperationCount = 0;
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            for(int d = 0; d < agent->degree; d++)
            {
                agent->setNewState_unique(agent->computeState_unique(d), d);
            }
            agent->computeReward_unique();
            agent->putLastAction();
        }
    
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            agent->updateQValue_unique();
            
            if(neighborRound == CONSTRAIN &&  agent->degree < SIZE - 1)
            {
                agent->updateRewireQValue();
                agent->chooseIfRewire();
            }
        }
        
        if(neighborRound == CONSTRAIN)
        {
            for (auto it = network.begin(); it != network.end(); ++it)
            {
                Agent* agent = *it;
                
                if(agent->neighbors.empty())
                {
                    continue;
                }
                
                if(agent->degree < SIZE - 1)
                {
                    list<Agent*> tempList = agent->neighbors;
                    agent->rewire(network, tempList);
                }
                
            }
            
            for (auto it = network.begin(); it != network.end(); ++it)
            {
                Agent* agent = *it;
                if(agent->neighbors.empty())
                {
                    continue;
                }

                agent->gotRewired();
            }
            
            neighborRound = 0;
        }
        
        for (auto it = network.begin(); it != network.end(); ++it)
        {
            Agent* agent = *it;
            
            if(agent->neighbors.empty())
            {
                continue;
            }

            agent->chooseAction_unique();
            agent->setCurrentState_unique();
            
            for(int d = 0; d < agent->degree; d++)
            {
                if(agent->getNewAction(d) == COOPERATE)
                {
                    currentCooperationCount += 1;
                    cooperCnt += 1;
                }
            }
        }
        
        double currentCooperationRate = (currentCooperationCount) / ((SIZE) * 4);
        cooperationRates.push_back(currentCooperationRate);
    }

    std::cout << "\n";
    std::cout << "Dr: " << Dr
              << " RC: " << CONSTRAIN
              << " Result: " << ((cooperCnt))/focusRound/((SIZE) * 4)
              << " Total rewire count: " << rewireCnt;
    
    int totalDegree2nd = 0;
    
    for (auto it = network.begin(); it != network.end(); ++it)
    {
        Agent* agent = *it;
        
        totalDegree2nd += agent->degree;
    }
    
    cout << "Node count: " << SIZE << "\n";
    cout << "Total degree sum: " << totalDegree2nd << "\n";
    cout << "Total rewire count: " << rewireCnt << "\n";
    cout << "Total got-rewired count: " << gotRewiredCnt << "\n";
    cout << "Rewire round threshold (CONSTRAIN): " << CONSTRAIN << "\n";
    cout << "Rewire cost: " << COST << "\n";
    cout << "Total rounds: " << totalRound << "\n";
    cout << "Focus rounds: " << focusRound << "\n";

    
    for (const auto& rate : cooperationRates)
    {
        outputFile << rate << "\n";
    }
    outputFile.close();
            
}

int main()
{
    sgenrand(SEED);

    PDG pdg;
    pdg.runGame();

    return 0;
}
