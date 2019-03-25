#!/usr/bin/env bash
# highest_attribute_value
python recommender_experiments.py --input_file 'data/input/cities/Charlotte.json' --recommender_algorithm 1 --output_dir 'data/output/' --clustering_algorithm 0 --configs 'checkins 2.0' 'checkins 2.042105263157895' 'checkins 2.0842105263157897' 'checkins 2.1263157894736846' 'checkins 2.1684210526315795' 'checkins 2.2105263157894743' 'checkins 2.252631578947369' 'checkins 2.294736842105264' 'checkins 2.336842105263159' 'checkins 2.378947368421054' 'checkins 2.4210526315789487' 'checkins 2.4631578947368435' 'checkins 2.5052631578947384' 'checkins 2.5473684210526333' 'checkins 2.589473684210528' 'checkins 2.631578947368423' 'checkins 2.673684210526318' 'checkins 2.7157894736842128' 'checkins 2.7578947368421076' 'checkins 2.8000000000000025' 'checkins 2.8421052631578974' 'checkins 2.8842105263157922' 'checkins 2.926315789473687' 'checkins 2.968421052631582' 'checkins 3.010526315789477' 'checkins 3.0526315789473717' 'checkins 3.0947368421052666' 'checkins 3.1368421052631614' 'checkins 3.1789473684210563' 'checkins 3.221052631578951' 'checkins 3.263157894736846' 'checkins 3.305263157894741' 'checkins 3.3473684210526358' 'checkins 3.3894736842105306' 'checkins 3.4315789473684255' 'checkins 3.4736842105263204' 'checkins 3.5157894736842152' 'checkins 3.55789473684211' 'checkins 3.600000000000005' 'checkins 3.6421052631579' 'checkins 3.6842105263157947' 'checkins 3.7263157894736896' 'checkins 3.7684210526315844' 'checkins 3.8105263157894793' 'checkins 3.852631578947374' 'checkins 3.894736842105269' 'checkins 3.936842105263164' 'checkins 3.978947368421059' 'checkins 4.021052631578954' 'checkins 4.0631578947368485' 'checkins 4.105263157894743' 'checkins 4.147368421052638' 'checkins 4.189473684210533' 'checkins 4.231578947368428' 'checkins 4.273684210526323' 'checkins 4.315789473684218' 'checkins 4.357894736842113' 'checkins 4.4000000000000075' 'checkins 4.442105263157902' 'checkins 4.484210526315797' 'checkins 4.526315789473692' 'checkins 4.568421052631587' 'checkins 4.610526315789482' 'checkins 4.652631578947377' 'checkins 4.6947368421052715' 'checkins 4.736842105263166' 'checkins 4.778947368421061' 'checkins 4.821052631578956' 'checkins 4.863157894736851' 'checkins 4.905263157894746' 'checkins 4.947368421052641' 'checkins 4.989473684210536' 'checkins 5.0315789473684305' 'checkins 5.073684210526325' 'checkins 5.11578947368422' 'checkins 5.157894736842115' 'checkins 5.20000000000001' 'checkins 5.242105263157905' 'checkins 5.2842105263158' 'checkins 5.3263157894736946' 'checkins 5.368421052631589' 'checkins 5.410526315789484' 'checkins 5.452631578947379' 'checkins 5.494736842105274' 'checkins 5.536842105263169' 'checkins 5.578947368421064' 'checkins 5.621052631578959' 'checkins 5.6631578947368535' 'checkins 5.705263157894748' 'checkins 5.747368421052643' 'checkins 5.789473684210538' 'checkins 5.831578947368433' 'checkins 5.873684210526328' 'checkins 5.915789473684223' 'checkins 5.957894736842118' 'checkins 6.000000000000012' 'checkins 6.042105263157907' 'checkins 6.084210526315802' 'checkins 6.126315789473697' 'checkins 6.168421052631592' 'checkins 6.210526315789487' 'checkins 6.252631578947382' 'checkins 6.2947368421052765' 'checkins 6.336842105263171' 'checkins 6.378947368421066' 'checkins 6.421052631578961' 'checkins 6.463157894736856' 'checkins 6.505263157894751' 'checkins 6.547368421052646' 'checkins 6.589473684210541' 'checkins 6.6315789473684355' 'checkins 6.67368421052633' 'checkins 6.715789473684225' 'checkins 6.75789473684212' 'checkins 6.800000000000015' 'checkins 6.84210526315791' 'checkins 6.884210526315805' 'checkins 6.9263157894736995' 'checkins 6.968421052631594' 'checkins 7.010526315789489' 'checkins 7.052631578947384' 'checkins 7.094736842105279' 'checkins 7.136842105263174' 'checkins 7.178947368421069' 'checkins 7.221052631578964' 'checkins 7.2631578947368585' 'checkins 7.305263157894753' 'checkins 7.347368421052648' 'checkins 7.389473684210543' 'checkins 7.431578947368438' 'checkins 7.473684210526333' 'checkins 7.515789473684228' 'checkins 7.5578947368421225' 'checkins 7.600000000000017' 'checkins 7.642105263157912' 'checkins 7.684210526315807' 'checkins 7.726315789473702' 'checkins 7.768421052631597' 'checkins 7.810526315789492' 'checkins 7.852631578947387' 'checkins 7.8947368421052815' 'checkins 7.936842105263176' 'checkins 7.978947368421071' 'checkins 8.021052631578966' 'checkins 8.063157894736861' 'checkins 8.105263157894756' 'checkins 8.14736842105265' 'checkins 8.189473684210546' 'checkins 8.23157894736844' 'checkins 8.273684210526335' 'checkins 8.31578947368423' 'checkins 8.357894736842125' 'checkins 8.40000000000002' 'checkins 8.442105263157915' 'checkins 8.48421052631581' 'checkins 8.526315789473704' 'checkins 8.5684210526316' 'checkins 8.610526315789494' 'checkins 8.652631578947389' 'checkins 8.694736842105284' 'checkins 8.736842105263179' 'checkins 8.778947368421074' 'checkins 8.821052631578969' 'checkins 8.863157894736863' 'checkins 8.905263157894758' 'checkins 8.947368421052653' 'checkins 8.989473684210548' 'checkins 9.031578947368443' 'checkins 9.073684210526338' 'checkins 9.115789473684233' 'checkins 9.157894736842128' 'checkins 9.200000000000022' 'checkins 9.242105263157917' 'checkins 9.284210526315812' 'checkins 9.326315789473707' 'checkins 9.368421052631602' 'checkins 9.410526315789497' 'checkins 9.452631578947392' 'checkins 9.494736842105286' 'checkins 9.536842105263181' 'checkins 9.578947368421076' 'checkins 9.621052631578971' 'checkins 9.663157894736866' 'checkins 9.70526315789476' 'checkins 9.747368421052656' 'checkins 9.78947368421055' 'checkins 9.831578947368445' 'checkins 9.87368421052634' 'checkins 9.915789473684235' 'checkins 9.95789473684213'
python recommender_experiments.py --input_file 'data/input/cities/Charlotte.json' --recommender_algorithm 1 --output_dir 'data/output/' --clustering_algorithm 1 --configs '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' '42' '43' '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'  '101' '102' '103' '104' '105' '106' '107' '108' '109' '110' '111' '112' '113' '114' '115' '116' '117' '118' '119' '120' '121' '122' '123' '124' '125' '126' '127' '128' '129' '130' '131' '132' '133' '134' '135' '136' '137' '138' '139' '140' '141' '142' '143' '144' '145' '146' '147' '148' '149' '150' '151' '152' '153' '154' '155' '156' '157' '158' '159' '160' '161' '162' '163' '164' '165' '166' '167' '168' '169' '170' '171' '172' '173' '174' '175' '176' '177' '178' '179' '180' '181' '182' '183' '184' '185' '186' '187' '188' '189' '190' '191' '192' '193' '194' '195' '196' '197' '198' '199' '200'
python recommender_experiments.py --input_file 'data/input/cities/Charlotte.json' --recommender_algorithm 1 --output_dir 'data/output/' --clustering_algorithm 3 --configs '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' '42' '43' '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'  '101' '102' '103' '104' '105' '106' '107' '108' '109' '110' '111' '112' '113' '114' '115' '116' '117' '118' '119' '120' '121' '122' '123' '124' '125' '126' '127' '128' '129' '130' '131' '132' '133' '134' '135' '136' '137' '138' '139' '140' '141' '142' '143' '144' '145' '146' '147' '148' '149' '150' '151' '152' '153' '154' '155' '156' '157' '158' '159' '160' '161' '162' '163' '164' '165' '166' '167' '168' '169' '170' '171' '172' '173' '174' '175' '176' '177' '178' '179' '180' '181' '182' '183' '184' '185' '186' '187' '188' '189' '190' '191' '192' '193' '194' '195' '196' '197' '198' '199' '200'

#python recommender_experiments.py --input_file 'data/input/cities/Las Vegas.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 0 --configs 'checkins 5.0' 'checkins 4.981' 'checkins 4.962' 'checkins 4.943' 'checkins 4.9239999999999995' 'checkins 4.904999999999999' 'checkins 4.885999999999999' 'checkins 4.866999999999999' 'checkins 4.847999999999999' 'checkins 4.828999999999999' 'checkins 4.809999999999999' 'checkins 4.790999999999999' 'checkins 4.7719999999999985' 'checkins 4.752999999999998' 'checkins 4.733999999999998' 'checkins 4.714999999999998' 'checkins 4.695999999999998' 'checkins 4.676999999999998' 'checkins 4.657999999999998' 'checkins 4.638999999999998' 'checkins 4.619999999999997' 'checkins 4.600999999999997' 'checkins 4.581999999999997' 'checkins 4.562999999999997' 'checkins 4.543999999999997' 'checkins 4.524999999999997' 'checkins 4.505999999999997' 'checkins 4.4869999999999965' 'checkins 4.467999999999996' 'checkins 4.448999999999996' 'checkins 4.429999999999996' 'checkins 4.410999999999996' 'checkins 4.391999999999996' 'checkins 4.372999999999996' 'checkins 4.353999999999996' 'checkins 4.3349999999999955' 'checkins 4.315999999999995' 'checkins 4.296999999999995' 'checkins 4.277999999999995' 'checkins 4.258999999999995' 'checkins 4.239999999999995' 'checkins 4.220999999999995' 'checkins 4.201999999999995' 'checkins 4.1829999999999945' 'checkins 4.163999999999994' 'checkins 4.144999999999994' 'checkins 4.125999999999994' 'checkins 4.106999999999994' 'checkins 4.087999999999994' 'checkins 4.068999999999994' 'checkins 4.049999999999994' 'checkins 4.0309999999999935' 'checkins 4.011999999999993' 'checkins 3.992999999999993' 'checkins 3.973999999999993' 'checkins 3.954999999999993' 'checkins 3.935999999999993' 'checkins 3.9169999999999927' 'checkins 3.8979999999999926' 'checkins 3.8789999999999925' 'checkins 3.8599999999999923' 'checkins 3.840999999999992' 'checkins 3.821999999999992' 'checkins 3.802999999999992' 'checkins 3.783999999999992' 'checkins 3.7649999999999917' 'checkins 3.7459999999999916' 'checkins 3.7269999999999914' 'checkins 3.7079999999999913' 'checkins 3.688999999999991' 'checkins 3.669999999999991' 'checkins 3.650999999999991' 'checkins 3.631999999999991' 'checkins 3.6129999999999907' 'checkins 3.5939999999999905' 'checkins 3.5749999999999904' 'checkins 3.5559999999999903' 'checkins 3.53699999999999' 'checkins 3.51799999999999' 'checkins 3.49899999999999' 'checkins 3.4799999999999898' 'checkins 3.4609999999999896' 'checkins 3.4419999999999895' 'checkins 3.4229999999999894' 'checkins 3.4039999999999893' 'checkins 3.384999999999989' 'checkins 3.365999999999989' 'checkins 3.346999999999989' 'checkins 3.3279999999999887' 'checkins 3.3089999999999886' 'checkins 3.2899999999999885' 'checkins 3.2709999999999884' 'checkins 3.2519999999999882' 'checkins 3.232999999999988' 'checkins 3.213999999999988' 'checkins 3.194999999999988' 'checkins 3.1759999999999877' 'checkins 3.1569999999999876' 'checkins 3.1379999999999875' 'checkins 3.1189999999999873'
#python recommender_experiments.py --input_file 'data/input/cities/Las Vegas.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 1 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'
#python recommender_experiments.py --input_file 'data/input/cities/Las Vegas.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 3 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'
#
#python recommender_experiments.py --input_file 'data/input/cities/Phoenix.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 0 --configs 'checkins 6.1' 'checkins 6.077999999999999' 'checkins 6.055999999999999' 'checkins 6.033999999999999' 'checkins 6.011999999999999' 'checkins 5.989999999999998' 'checkins 5.967999999999998' 'checkins 5.945999999999998' 'checkins 5.923999999999998' 'checkins 5.9019999999999975' 'checkins 5.879999999999997' 'checkins 5.857999999999997' 'checkins 5.835999999999997' 'checkins 5.8139999999999965' 'checkins 5.791999999999996' 'checkins 5.769999999999996' 'checkins 5.747999999999996' 'checkins 5.7259999999999955' 'checkins 5.703999999999995' 'checkins 5.681999999999995' 'checkins 5.659999999999995' 'checkins 5.637999999999995' 'checkins 5.615999999999994' 'checkins 5.593999999999994' 'checkins 5.571999999999994' 'checkins 5.549999999999994' 'checkins 5.527999999999993' 'checkins 5.505999999999993' 'checkins 5.483999999999993' 'checkins 5.461999999999993' 'checkins 5.439999999999992' 'checkins 5.417999999999992' 'checkins 5.395999999999992' 'checkins 5.373999999999992' 'checkins 5.351999999999991' 'checkins 5.329999999999991' 'checkins 5.307999999999991' 'checkins 5.285999999999991' 'checkins 5.2639999999999905' 'checkins 5.24199999999999' 'checkins 5.21999999999999' 'checkins 5.19799999999999' 'checkins 5.1759999999999895' 'checkins 5.153999999999989' 'checkins 5.131999999999989' 'checkins 5.109999999999989' 'checkins 5.0879999999999885' 'checkins 5.065999999999988' 'checkins 5.043999999999988' 'checkins 5.021999999999988' 'checkins 4.999999999999988' 'checkins 4.977999999999987' 'checkins 4.955999999999987' 'checkins 4.933999999999987' 'checkins 4.911999999999987' 'checkins 4.889999999999986' 'checkins 4.867999999999986' 'checkins 4.845999999999986' 'checkins 4.823999999999986' 'checkins 4.801999999999985' 'checkins 4.779999999999985' 'checkins 4.757999999999985' 'checkins 4.735999999999985' 'checkins 4.713999999999984' 'checkins 4.691999999999984' 'checkins 4.669999999999984' 'checkins 4.647999999999984' 'checkins 4.6259999999999835' 'checkins 4.603999999999983' 'checkins 4.581999999999983' 'checkins 4.559999999999983' 'checkins 4.5379999999999825' 'checkins 4.515999999999982' 'checkins 4.493999999999982' 'checkins 4.471999999999982' 'checkins 4.4499999999999815' 'checkins 4.427999999999981' 'checkins 4.405999999999981' 'checkins 4.383999999999981' 'checkins 4.361999999999981' 'checkins 4.33999999999998' 'checkins 4.31799999999998' 'checkins 4.29599999999998' 'checkins 4.27399999999998' 'checkins 4.251999999999979' 'checkins 4.229999999999979' 'checkins 4.207999999999979' 'checkins 4.185999999999979' 'checkins 4.163999999999978' 'checkins 4.141999999999978' 'checkins 4.119999999999978' 'checkins 4.097999999999978' 'checkins 4.075999999999977' 'checkins 4.053999999999977' 'checkins 4.031999999999977' 'checkins 4.009999999999977' 'checkins 3.9879999999999765' 'checkins 3.965999999999976' 'checkins 3.943999999999976' 'checkins 3.9219999999999757'
#python recommender_experiments.py --input_file 'data/input/cities/Phoenix.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 1 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'
#python recommender_experiments.py --input_file 'data/input/cities/Phoenix.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 3 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'
#
#python recommender_experiments.py --input_file 'data/input/cities/Pittsburgh.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 0 --configs 'checkins 3.5' 'checkins 3.4861' 'checkins 3.4722' 'checkins 3.4583' 'checkins 3.4444' 'checkins 3.4305' 'checkins 3.4166' 'checkins 3.4027' 'checkins 3.3888' 'checkins 3.3749' 'checkins 3.3609999999999998' 'checkins 3.3470999999999997' 'checkins 3.3331999999999997' 'checkins 3.3192999999999997' 'checkins 3.3053999999999997' 'checkins 3.2914999999999996' 'checkins 3.2775999999999996' 'checkins 3.2636999999999996' 'checkins 3.2497999999999996' 'checkins 3.2358999999999996' 'checkins 3.2219999999999995' 'checkins 3.2080999999999995' 'checkins 3.1941999999999995' 'checkins 3.1802999999999995' 'checkins 3.1663999999999994' 'checkins 3.1524999999999994' 'checkins 3.1385999999999994' 'checkins 3.1246999999999994' 'checkins 3.1107999999999993' 'checkins 3.0968999999999993' 'checkins 3.0829999999999993' 'checkins 3.0690999999999993' 'checkins 3.0551999999999992' 'checkins 3.0412999999999992' 'checkins 3.027399999999999' 'checkins 3.013499999999999' 'checkins 2.999599999999999' 'checkins 2.985699999999999' 'checkins 2.971799999999999' 'checkins 2.957899999999999' 'checkins 2.943999999999999' 'checkins 2.930099999999999' 'checkins 2.916199999999999' 'checkins 2.902299999999999' 'checkins 2.888399999999999' 'checkins 2.874499999999999' 'checkins 2.860599999999999' 'checkins 2.846699999999999' 'checkins 2.832799999999999' 'checkins 2.818899999999999' 'checkins 2.804999999999999' 'checkins 2.791099999999999' 'checkins 2.777199999999999' 'checkins 2.7632999999999988' 'checkins 2.7493999999999987' 'checkins 2.7354999999999987' 'checkins 2.7215999999999987' 'checkins 2.7076999999999987' 'checkins 2.6937999999999986' 'checkins 2.6798999999999986' 'checkins 2.6659999999999986' 'checkins 2.6520999999999986' 'checkins 2.6381999999999985' 'checkins 2.6242999999999985' 'checkins 2.6103999999999985' 'checkins 2.5964999999999985' 'checkins 2.5825999999999985' 'checkins 2.5686999999999984' 'checkins 2.5547999999999984' 'checkins 2.5408999999999984' 'checkins 2.5269999999999984' 'checkins 2.5130999999999983' 'checkins 2.4991999999999983' 'checkins 2.4852999999999983' 'checkins 2.4713999999999983' 'checkins 2.4574999999999982' 'checkins 2.443599999999998' 'checkins 2.429699999999998' 'checkins 2.415799999999998' 'checkins 2.401899999999998' 'checkins 2.387999999999998' 'checkins 2.374099999999998' 'checkins 2.360199999999998' 'checkins 2.346299999999998' 'checkins 2.332399999999998' 'checkins 2.318499999999998' 'checkins 2.304599999999998' 'checkins 2.290699999999998' 'checkins 2.276799999999998' 'checkins 2.262899999999998' 'checkins 2.248999999999998' 'checkins 2.235099999999998' 'checkins 2.221199999999998' 'checkins 2.207299999999998' 'checkins 2.193399999999998' 'checkins 2.1794999999999978' 'checkins 2.1655999999999977' 'checkins 2.1516999999999977' 'checkins 2.1377999999999977' 'checkins 2.1238999999999977'
#python recommender_experiments.py --input_file 'data/input/cities/Pittsburgh.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 1 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'
#python recommender_experiments.py --input_file 'data/input/cities/Pittsburgh.json' --recommender_algorithm 0 --output_dir 'data/output/' --clustering_algorithm 3 --configs '50' '51' '52' '53' '54' '55' '56' '57' '58' '59' '60' '61' '62' '63' '64' '65' '66' '67' '68' '69' '70' '71' '72' '73' '74' '75' '76' '77' '78' '79' '80' '81' '82' '83' '84' '85' '86' '87' '88' '89' '90' '91' '92' '93' '94' '95' '96' '97' '98' '99' '100'


