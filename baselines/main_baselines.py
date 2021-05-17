import sys
import pandas as pd

sys.path.append('../')
from cobs.model import Model
from default_config import eplus_naming_dict, eplus_var_types


def setup(ep_model, season, blinds, daylighting):
    print('running', season, blinds, daylighting)
    if daylighting:
        dlight = 1
    else:
        dlight = 0

    if season == 'heating':
        blind_type = 'OnNightIfLowOutdoorTempAndOffDay'
        run_period = (35, 1991, 1, 1)
    elif season == 'cooling':
        blind_type = 'OffNightAndOnDayIfCoolingAndHighSolarOnWindow'
        run_period = (35, 1991, 7, 1)
    else:
        raise ValueError(f'Incorrect value for season: {season}')

    # turn off if blinds are not active
    if not blinds:
        blind_type = 'AlwaysOff'

    # Set the run period for the defined season
    ep_model.set_runperiod(*run_period)

    if blinds:
        ep_model.set_blinds(
            ["WB-1", "WL-1", "WF-1", "WR-1"],
            blind_material_name="White Painted Metal Blind",
            shading_control_type=blind_type,
            agent_control=False
        )

    # Setup Daylighting
    ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'DaylightingAvail'}, {
        'Field 4': dlight
    })

    return ep_model, f'stptCntrl_{season}_blinds{blinds}_daylighting{daylighting}.csv'


def run(season, blinds, daylighting):
    ep_model = Model(idf_file_name='../eplus_files/5Zone_Control_Therm.idf',
                     eplus_naming_dict=eplus_naming_dict,
                     eplus_var_types=eplus_var_types,
                     weather_file='../eplus_files/pittsburg_TMY3.epw')

    ep_model, fname = setup(ep_model, season, blinds, daylighting)
    print('running', fname)
    i = 0
    # Run simulation
    state_history = []
    state = ep_model.reset()
    state_history.append(state)
    while not ep_model.is_terminate():
        state = ep_model.step(list())
        state_history.append(state)
        i += 1
    result = pd.DataFrame(state_history)
    result['time'] = result['time'].mask(result['time'].dt.year > 1,  # Warn: hacky way of replacing year
                                         result['time'] + pd.offsets.DateOffset(year=1991))
    df = result[list(eplus_naming_dict.values()) + ['time']].copy()
    df.index = df['time']
    df.to_csv(fname)
    return df


if __name__ == '__main__':
    print('\n Starting ...')
    Model.set_energyplus_folder("/Applications/EnergyPlus-9-3-0-bugfix/")

    run('heating', False, False)
    run('heating', True, False)
    run('cooling', False, False)
    run('cooling', True, False)
