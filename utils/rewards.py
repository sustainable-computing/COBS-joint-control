import pdb


class BaseReward:
    def r_func(self, obs_dict, action):
        return 0


# class ViolationPCoilReward(BaseReward):
#     def __init__(self, cooling_mult, heating_mult=0):
#         """
#
#         :param cooling_mult: Depends on scale of coil power
#         :param heating_mult:
#         """
#         self.cooling_mult = cooling_mult
#         self.heating_mult = heating_mult
#
#     def r_func(self, obs_dict, action):
#         T_upper = 24
#         T_lower = 19
#
#         temp = obs_dict["Indoor Temp."]
#         cost = obs_dict["Coil Power"]
#
#         # temp_violation = max(T_lower - temp, 0)
#         temp_violation = self.heating_mult * max(temp - T_upper, 0) + self.cooling_mult * max(T_lower - temp, 0)
#         rt = (-1 * cost) - temp_violation
#         return rt


class OctoReward(BaseReward):
    def __init__(self, power_mult, therm_mult, vis_mult, power_range, therm_range, vis_range, multi_agent):
        self.power_mult = power_mult
        self.therm_mult = therm_mult
        self.vis_mult = vis_mult
        self.power_range = power_range
        self.therm_range = therm_range  # TODO -0.5 from max and min
        self.vis_range = vis_range  # TODO - 500
        self.multi_agent = multi_agent

    @staticmethod
    def norm(x, range):
        return (x - range[0]) / (range[1] - range[0])

    def calc_e(self, obs):
        # Only include the lights from the occupied zones
        if not self.multi_agent:
            obs['Lights'] = obs['Lights Zone 1'] + \
                            obs['Lights Zone 2'] + \
                            obs['Lights Zone 3'] + \
                            obs['Lights Zone 4'] + \
                            obs['Lights Zone 5']
            return self.norm(obs["HVAC Power"] + obs['Lights'], self.power_range)

        results = list()
        for i in range(1, 6):
            results.append(
                self.norm(
                    obs[f'Lights Zone {i}'] + obs["HVAC Power"],
                    self.power_range))
        return results

    def calc_tc(self, obs):
        # ranges come from octopus paper
        if not self.multi_agent:
            pmv = 0
            zone_count = 0
            for i in range(1, 6):
                if obs['occupancy'][f'SPACE{i}-1'] != 0:
                    pmv += obs[f'PMV Zone {i}']
                    zone_count += 1

            if zone_count != 0:
                pmv /= zone_count

            if pmv < -0.5:
                e = abs(pmv + 0.5)
            elif pmv > 0.5:
                e = abs(pmv - 0.5)
            else:
                return 0
            return self.norm(e, self.therm_range)  # * int(obs['Occupancy Flag'])

        results = list()
        for i in range(1, 6):
            pmv = obs[f'PMV Zone {i}']
            if pmv < -0.5:
                e = abs(pmv + 0.5)
                results.append(self.norm(e, self.therm_range))
            elif pmv > 0.5:
                e = abs(pmv - 0.5)
                results.append(self.norm(e, self.therm_range))
            else:
                results.append(0)
        return results

    def calc_vc(self, obs):
        # https://www.archtoolbox.com/materials-systems/electrical/recommended-lighting-levels-in-buildings.html
        obs['Illum Zone 1'] = (obs['Illum 1 Zone 1'] + obs['Illum 2 Zone 1']) / 2
        obs['Illum Zone 3'] = (obs['Illum 1 Zone 3'] + obs['Illum 2 Zone 3']) / 2
        if not self.multi_agent:
            illum = 0
            zone_count = 0
            for i in range(1, 5):
                if obs['occupancy'][f'SPACE{i}-1'] != 0:
                    f = obs[f'Illum Zone {i}']
                    zone_count += 1
                    # if f < 300:
                    #     e = 300 - f
                    if f > 750:
                        e = f - 750
                    else:
                        e = 0
                    illum += e
            if zone_count != 0:
                illum /= zone_count
            return self.norm(illum, self.vis_range)  # * int(obs['occupancy'][f'SPACE1-1'] != 0)

        results = list()
        for i in range(1, 5):
            illum = 0
            if obs['occupancy'][f'SPACE{i}-1'] != 0:
                illum = obs[f'Illum Zone {i}']
                # if illum < 300:
                #     illum = 300 - illum
                if illum > 750:
                    illum = illum - 750
                else:
                    illum = 0
            results.append(self.norm(illum, self.vis_range))
        results.append(0)
        return results

    def reward(self, obs_dict, *args):
        e = self.calc_e(obs_dict)
        tc = self.calc_tc(obs_dict)
        vc = self.calc_vc(obs_dict)
        if not self.multi_agent:
            return [-1 * (self.power_mult * e + self.therm_mult * tc + self.vis_mult * vc)]

        results = list()
        for i in range(len(e)):
            results.append(-1 * (self.power_mult * e[i] + self.therm_mult * tc[i] + self.vis_mult * vc[i]))
        return results


class ViolationPCoilReward(BaseReward):
    def __init__(self, occ_weight):
        """

        :param cooling_mult: Depends on scale of coil power
        :param heating_mult:
        """
        self.occ_weight = occ_weight

    def reward(self, obs_dict, action):
        tot_power = obs_dict["Heat Coil Power"] + obs_dict["Cool Coil Power"]
        cost = tot_power / 3800
        ppd = obs_dict["PPD"] / 100
        occ = obs_dict["Occupancy Flag"]

        # temp_violation = max(T_lower - temp, 0)
        rt = - cost - ppd * int(occ) * self.occ_weight
        return rt


class ViolationPActionReward(BaseReward):
    def __init__(self, occ_weight, eta=[0.1, 4]):
        """ The reward from the Gnu-RL BuildSys Paper

        :param occ_weight: between 0 and 1
        :param eta: Weight for comfort during unoccupied and occupied mode
        """
        self.occ_weight = occ_weight
        self.eta = eta

    def reward(self, obs_dict, action):
        print(action)

        if "Schedule:Constant|*|Schedule Value|*|SAT_SP" in action["actuator"]:
            action = action["actuator"]["Schedule:Constant|*|Schedule Value|*|SAT_SP"][2]
        else:
            action = 0
        # action, SAT_stpt = action
        temp = obs_dict["Indoor Temp."]
        occ = obs_dict["Occupancy Flag"]
        stpt = obs_dict["Indoor Temp. Setpoint"]
        r = - self.occ_weight * self.eta[int(occ)] * (temp - stpt) ** 2 - action  # [0]
        return r

    # def reward(self, obs_dict, action):
    #     try:
    #         action, SAT_stpt = action
    #     except ValueError:
    #         action, SAT_stpt, _ = action
    #     temp = obs_dict["Indoor Temp."]
    #     occ = obs_dict["Occupancy Flag"]
    #     stpt = obs_dict["Indoor Temp. Setpoint"]
    #     r = - self.occ_weight * self.eta[int(occ)] * (temp - stpt) ** 2 - action#[0]
    #     return r


class PPDPActionReward(BaseReward):
    def __init__(self, occ_weight, eta=[0, 4]):
        """ The reward from the Gnu-RL BuildSys Paper

        :param occ_weight: between 0 and 1
        :param eta: Weight for comfort during unoccupied and occupied mode
        """
        self.occ_weight = occ_weight
        self.eta = eta

    def reward(self, obs_dict, action):
        try:
            action, SAT_stpt = action
        except ValueError:
            action, SAT_stpt, _ = action

        # temp = obs_dict["Indoor Temp."]
        occ = obs_dict["Occupancy Flag"]
        # stpt = obs_dict["Indoor Temp. Setpoint"]
        ppd = obs_dict["PPD"]
        r = - self.occ_weight * self.eta[int(occ)] * ppd - action ** 2  # [0]
        return r
