#include "mnist_digits_model.h"

conv_layer_t conv_layer0; // Capa convolucional 1
conv_layer_t conv_layer1; // Capa convolucional 2
dense_layer_t dense_layer0; // Capa densa 1
dense_layer_t dense_layer1; // Capa densa 2


conv_layer_t init_conv_layer0(void){

  static filter_t filtros[4];
  
  static const float weights0[]={
    -0.4422368109226227, -0.8453366756439209, -0.6311924457550049, 
    0.09957651048898697, -0.5449668169021606, -0.4683987498283386, 
    0.7659248113632202, -0.35742461681365967, 0.4306115210056305, 
  
  };
  static filter_t filter0 = {1, 3, weights0, -0.7063255310058594};
  filtros[0]=filter0;
    
  static const float weights1[]={
    0.4023783504962921, 0.5517668724060059, 0.6405840516090393, 
    -0.8286018967628479, 0.41422930359840393, 0.4170520305633545, 
    -0.5056888461112976, 0.1364106833934784, 0.6513766646385193, 
  
  };
  static filter_t filter1 = {1, 3, weights1, -0.44184693694114685};
  filtros[1]=filter1;
    
  static const float weights2[]={
    0.6207616329193115, 0.4769766628742218, 0.45333099365234375, 
    -0.023933468386530876, 0.5333149433135986, -0.018755488097667694, 
    -0.0379830002784729, -0.8465821743011475, -0.6245033740997314, 
  
  };
  static filter_t filter2 = {1, 3, weights2, -0.24197842180728912};
  filtros[2]=filter2;
    
  static const float weights3[]={
    0.055423833429813385, -0.8854941129684448, -0.9101300835609436, 
    0.2282666116952896, 0.7109596133232117, -0.1472686231136322, 
    0.12485599517822266, 0.2962336242198944, 0.9329885244369507, 
  
  };
  static filter_t filter3 = {1, 3, weights3, -0.18503688275814056};
  filtros[3]=filter3;
    
  conv_layer_t layer = {4,filtros};
  return layer;
}
  

conv_layer_t init_conv_layer1(void){

  static filter_t filtros[8];
  
  static const float weights0[]={
    -0.2740195393562317, -0.36819449067115784, -0.011240936815738678, 
    0.3359537422657013, 0.24846859276294708, -0.22152383625507355, 
    0.6899257898330688, -0.0398431234061718, -0.07440643757581711, 
  
    -0.17267124354839325, 0.039878830313682556, 0.5038106441497803, 
    -0.28503188490867615, 0.0407976433634758, -0.34954673051834106, 
    0.22045622766017914, -0.3508385419845581, -0.43785256147384644, 
  
    0.9560909867286682, -0.033294398337602615, 0.31957730650901794, 
    -0.3016243278980255, -0.24100810289382935, 0.10208050906658173, 
    -0.2552366852760315, 0.40453100204467773, -0.16340313851833344, 
  
    -0.278658002614975, -0.3097078204154968, -0.3943988084793091, 
    0.16891548037528992, 0.06351436674594879, 0.37358319759368896, 
    -0.08834774792194366, -0.21253475546836853, 0.36459872126579285, 
  
  };
  static filter_t filter0 = {4, 3, weights0, -0.02448587864637375};
  filtros[0]=filter0;
    
  static const float weights1[]={
    -0.00663350522518158, -0.5404090881347656, 0.3888988792896271, 
    0.06088811159133911, 0.38891980051994324, -0.3361142873764038, 
    -0.46659696102142334, 0.1384451687335968, 0.30239030718803406, 
  
    -0.13783632218837738, 0.07002408057451248, 0.6139779090881348, 
    -0.24713034927845, -0.19832642376422882, 0.2246813029050827, 
    0.5253620147705078, -0.5656241178512573, -0.20024940371513367, 
  
    0.1782844215631485, -0.20190685987472534, -0.6227609515190125, 
    0.2379893660545349, -0.10858981311321259, 0.011483857408165932, 
    -0.46087491512298584, 0.24655000865459442, -0.3609243929386139, 
  
    0.3755761384963989, 0.5851845145225525, 0.1665639728307724, 
    -0.11687822639942169, -0.42401742935180664, -0.2699694335460663, 
    0.022285394370555878, -0.08614718914031982, 0.6928908824920654, 
  
  };
  static filter_t filter1 = {4, 3, weights1, -0.05836941674351692};
  filtros[1]=filter1;
    
  static const float weights2[]={
    -0.08855665475130081, 0.004512117709964514, -0.11759759485721588, 
    0.27673378586769104, -0.256788969039917, -0.09912613034248352, 
    0.8730527758598328, 0.2538200616836548, -0.13210053741931915, 
  
    -0.2585373818874359, 0.39240899682044983, -0.1123931035399437, 
    -0.37830644845962524, 0.16082656383514404, 0.2533971965312958, 
    -0.48819607496261597, 0.4495560824871063, 0.43600934743881226, 
  
    0.7301000952720642, -0.21410177648067474, -0.2572496831417084, 
    0.007549460045993328, -0.2880033254623413, -0.45417651534080505, 
    0.5628681182861328, -0.5384514331817627, -0.23049700260162354, 
  
    -0.4002463221549988, -0.4318258762359619, -0.1559952050447464, 
    0.08720401674509048, 0.004491634666919708, -0.29375791549682617, 
    0.37241458892822266, 0.5714399814605713, 1.1291099786758423, 
  
  };
  static filter_t filter2 = {4, 3, weights2, -0.17405006289482117};
  filtros[2]=filter2;
    
  static const float weights3[]={
    0.24516047537326813, -0.015445161610841751, 0.03661813959479332, 
    -0.010202718898653984, 0.39786940813064575, 0.18385085463523865, 
    -0.08773206174373627, 0.22644799947738647, -0.31000012159347534, 
  
    0.06791907548904419, -0.08861526101827621, 0.3692568838596344, 
    0.06617072969675064, -0.1380661129951477, 0.37508219480514526, 
    0.1142987310886383, -0.04697151854634285, -0.37280434370040894, 
  
    0.46503302454948425, 0.09820310026407242, 0.5984407067298889, 
    -0.2207411527633667, -0.20580507814884186, -0.16048988699913025, 
    0.026181256398558617, -0.06940346956253052, 0.261347234249115, 
  
    -7.232051575556397e-05, 0.05415261909365654, 0.04721590504050255, 
    0.12118826061487198, 0.5097700357437134, 1.1849682331085205, 
    0.19040659070014954, 0.12893874943256378, 0.06711817532777786, 
  
  };
  static filter_t filter3 = {4, 3, weights3, -0.005460951942950487};
  filtros[3]=filter3;
    
  static const float weights4[]={
    0.4377250671386719, -0.08576484769582748, 0.03856559842824936, 
    0.5657821297645569, -0.22078576683998108, -0.14287538826465607, 
    -0.4303893744945526, -0.45777323842048645, 0.3564847707748413, 
  
    -0.3578803241252899, 0.18207551538944244, -0.22379060089588165, 
    -0.10820206254720688, -0.2670046389102936, -0.43136006593704224, 
    -0.018013358116149902, -0.08876723051071167, 0.49399712681770325, 
  
    -0.3008798658847809, 0.06378183513879776, -0.14134381711483002, 
    -0.7438258528709412, 0.04609796777367592, 0.09631982445716858, 
    -0.45134785771369934, -0.23986871540546417, -0.17009475827217102, 
  
    0.2523399889469147, -0.00927820522338152, -0.03497244045138359, 
    0.07232201844453812, 0.3015184700489044, -0.15034426748752594, 
    -0.7069684863090515, 0.15383529663085938, 1.1667734384536743, 
  
  };
  static filter_t filter4 = {4, 3, weights4, 3.995356382802129e-05};
  filtros[4]=filter4;
    
  static const float weights5[]={
    -0.39393848180770874, -0.9724516868591309, -0.2790030837059021, 
    0.021366486325860023, -0.16902390122413635, 0.1523294746875763, 
    0.5291719436645508, 0.11814236640930176, 0.39178091287612915, 
  
    0.10114612430334091, 0.604396641254425, 0.5160820484161377, 
    -0.29295027256011963, 0.23383629322052002, -0.30351316928863525, 
    -0.22484752535820007, -0.06053346022963524, 0.0630059763789177, 
  
    0.25720760226249695, 0.08735989779233932, 0.34954357147216797, 
    0.33966636657714844, 0.5142858028411865, 0.039937574416399, 
    -0.40483108162879944, -0.3452852964401245, -0.02564426138997078, 
  
    -0.1472543329000473, 0.08040685206651688, -0.5607771277427673, 
    0.36733871698379517, -0.12385564297437668, 0.034629493951797485, 
    -0.2891704738140106, 0.040802597999572754, 0.43644633889198303, 
  
  };
  static filter_t filter5 = {4, 3, weights5, 0.1654168665409088};
  filtros[5]=filter5;
    
  static const float weights6[]={
    0.3199171721935272, -0.564228892326355, 0.21922199428081512, 
    0.06976543366909027, -0.4305315315723419, -0.07674862444400787, 
    -0.0030583934858441353, 0.4209350645542145, 0.16236397624015808, 
  
    0.18265649676322937, 0.012350903823971748, 0.5097543001174927, 
    0.025476813316345215, -0.18551215529441833, 0.007555271033197641, 
    -0.39503777027130127, -0.11555664241313934, 0.05515997111797333, 
  
    0.05212017893791199, -0.34273213148117065, -0.2797299027442932, 
    -0.019094571471214294, -0.4809977412223816, 0.519439697265625, 
    0.358585923910141, 0.5740707516670227, 0.5519087910652161, 
  
    0.5204744338989258, -0.22200292348861694, 0.8627483248710632, 
    0.48580366373062134, 0.1579999029636383, -0.6209015846252441, 
    0.5470477342605591, 0.10231857001781464, -0.19988159835338593, 
  
  };
  static filter_t filter6 = {4, 3, weights6, 0.41553330421447754};
  filtros[6]=filter6;
    
  static const float weights7[]={
    0.24969646334648132, -0.18154498934745789, -0.1398598849773407, 
    -0.1495216339826584, 0.25309887528419495, -0.6801113486289978, 
    -0.43701544404029846, 0.038560375571250916, -0.34399163722991943, 
  
    -0.4677129089832306, 0.48752161860466003, -0.25742682814598083, 
    -0.09929665923118591, 0.19700130820274353, -0.09692174941301346, 
    0.314163476228714, 0.6544336676597595, -0.3668915331363678, 
  
    0.13796542584896088, -0.06458538770675659, -0.19157877564430237, 
    0.13970494270324707, -0.554498553276062, 0.21727454662322998, 
    0.34053218364715576, 0.6577886343002319, 0.05401527136564255, 
  
    -0.15368130803108215, 0.004355575889348984, -0.2707130014896393, 
    -0.41539353132247925, -0.766722559928894, 0.29697731137275696, 
    -0.29688242077827454, -0.5257664918899536, -0.4552977681159973, 
  
  };
  static filter_t filter7 = {4, 3, weights7, -0.02466008812189102};
  filtros[7]=filter7;
    
  conv_layer_t layer = {8,filtros};
  return layer;
}
  
dense_layer_t init_dense_layer0(){
  // Cantidad de variables weights = numero de neuronas
  // Cantidad de pesos por weights = numero de entradas

  static neuron_t neuronas[16];
  
  static const float weights0[]={
    -0.6389639377593994, 0.8738739490509033, 1.0141093730926514, -0.034705255180597305, 1.8000563383102417, 0.5990874171257019, 0.47677189111709595, -0.8369766473770142,
  };
  static neuron_t neuron0 = {weights0, 0.14565719664096832};
  neuronas[0]=neuron0;
    
  static const float weights1[]={
    -0.0799064114689827, 0.20690979063510895, -0.6488179564476013, 1.3101974725723267, -0.6971604228019714, -1.5539789199829102, -0.45433637499809265, -0.6804951429367065,
  };
  static neuron_t neuron1 = {weights1, -0.27145111560821533};
  neuronas[1]=neuron1;
    
  static const float weights2[]={
    -0.014509722590446472, 1.0842481851577759, -0.3308371305465698, -0.5328399538993835, -0.4857247471809387, -0.29367387294769287, 1.6087062358856201, 0.28030601143836975,
  };
  static neuron_t neuron2 = {weights2, 0.32751500606536865};
  neuronas[2]=neuron2;
    
  static const float weights3[]={
    -0.40904372930526733, -0.2730678915977478, -0.8189523816108704, 0.02597367763519287, -1.6772499084472656, -0.1888471394777298, 0.23301801085472107, 1.2201364040374756,
  };
  static neuron_t neuron3 = {weights3, 0.13192419707775116};
  neuronas[3]=neuron3;
    
  static const float weights4[]={
    0.45475804805755615, 0.5619512796401978, -0.29521939158439636, -1.040402889251709, 0.9088175296783447, 1.2532069683074951, -0.3462073802947998, 0.3889319896697998,
  };
  static neuron_t neuron4 = {weights4, -0.025017958134412766};
  neuronas[4]=neuron4;
    
  static const float weights5[]={
    -0.3019864857196808, -1.7942304611206055, -1.0968027114868164, -0.6195197105407715, 0.47029542922973633, -0.39073094725608826, -0.4965234398841858, 0.40580594539642334,
  };
  static neuron_t neuron5 = {weights5, 0.15264031291007996};
  neuronas[5]=neuron5;
    
  static const float weights6[]={
    1.3023570775985718, -0.6037102341651917, 0.29174545407295227, 1.5953158140182495, -0.26001864671707153, 0.1951490342617035, -0.9614193439483643, 0.17451262474060059,
  };
  static neuron_t neuron6 = {weights6, -0.06346044689416885};
  neuronas[6]=neuron6;
    
  static const float weights7[]={
    0.39416223764419556, 0.08426859229803085, 0.9767055511474609, -0.7320954203605652, 1.137419581413269, 0.14005254209041595, -0.7202475070953369, 0.8265734314918518,
  };
  static neuron_t neuron7 = {weights7, 0.327996164560318};
  neuronas[7]=neuron7;
    
  static const float weights8[]={
    0.6164255142211914, 1.617773413658142, -0.9425971508026123, 0.12925346195697784, 1.013684868812561, -0.05079317465424538, 0.361758291721344, -0.5659061670303345,
  };
  static neuron_t neuron8 = {weights8, -0.2686346769332886};
  neuronas[8]=neuron8;
    
  static const float weights9[]={
    0.340359091758728, 0.6281698942184448, -0.9565685987472534, 0.6253677606582642, -0.12502576410770416, -1.2071586847305298, 0.595542311668396, 0.5235103368759155,
  };
  static neuron_t neuron9 = {weights9, 0.0019406351493671536};
  neuronas[9]=neuron9;
    
  static const float weights10[]={
    -0.14366690814495087, -0.043678563088178635, -0.31323665380477905, 0.889727771282196, 0.46510788798332214, -1.262236475944519, -0.5223695635795593, 1.2070016860961914,
  };
  static neuron_t neuron10 = {weights10, -0.058686308562755585};
  neuronas[10]=neuron10;
    
  static const float weights11[]={
    0.2306605726480484, 0.3359133005142212, 1.3087884187698364, 0.35357242822647095, -0.9232487082481384, 0.6566071510314941, -0.3104857802391052, -1.024158000946045,
  };
  static neuron_t neuron11 = {weights11, -0.22518326342105865};
  neuronas[11]=neuron11;
    
  static const float weights12[]={
    1.3144234418869019, -0.4033755660057068, 0.901322603225708, 0.6902052760124207, 0.32145678997039795, 0.6292345523834229, 0.09172851592302322, 1.1400020122528076,
  };
  static neuron_t neuron12 = {weights12, 0.07922948896884918};
  neuronas[12]=neuron12;
    
  static const float weights13[]={
    -0.7116009593009949, -1.0916513204574585, 1.2796257734298706, -0.7936440110206604, -0.27963244915008545, -0.13555370271205902, 0.74143385887146, 0.8813574314117432,
  };
  static neuron_t neuron13 = {weights13, 0.03383558616042137};
  neuronas[13]=neuron13;
    
  static const float weights14[]={
    0.7083987593650818, 0.5162839889526367, 0.1731940507888794, 0.7994467616081238, -0.45469096302986145, 1.2522344589233398, 1.065606951713562, -0.6927228569984436,
  };
  static neuron_t neuron14 = {weights14, 0.11591171473264694};
  neuronas[14]=neuron14;
    
  static const float weights15[]={
    -0.258116215467453, -0.9853728413581848, -0.009293165057897568, 0.97235506772995, -0.5219270586967468, -0.3981132209300995, 1.5941957235336304, -1.188555359840393,
  };
  static neuron_t neuron15 = {weights15, -0.23319824039936066};
  neuronas[15]=neuron15;
    
  dense_layer_t layer= {16, neuronas};
  return layer;
}

dense_layer_t init_dense_layer1(){
  // Cantidad de variables weights = numero de neuronas
  // Cantidad de pesos por weights = numero de entradas

  static neuron_t neuronas[10];
  
  static const float weights0[]={
    1.0587925910949707, 1.513569712638855, 1.3243600130081177, 0.6540852189064026, -0.8047923445701599, -1.4307787418365479, -0.33441850543022156, -0.2860252261161804, 1.672497034072876, 1.365515947341919, -0.019888250157237053, 0.36380431056022644, -0.5840712785720825, -1.4365735054016113, 0.1887657195329666, 1.0845757722854614,
  };
  static neuron_t neuron0 = {weights0, -0.20080997049808502};
  neuronas[0]=neuron0;
    
  static const float weights1[]={
    0.7612287998199463, -1.1616497039794922, -0.8461942672729492, 0.7101844549179077, 1.3521677255630493, 1.4899402856826782, -1.1485133171081543, 1.2580504417419434, -0.9422171711921692, -0.5374773740768433, -0.21304640173912048, -0.24495162069797516, -0.2509096562862396, 0.7129276990890503, -1.953637957572937, -2.101435422897339,
  };
  static neuron_t neuron1 = {weights1, 0.15499626100063324};
  neuronas[1]=neuron1;
    
  static const float weights2[]={
    1.0468404293060303, -0.331094354391098, 0.4403402805328369, -1.1296151876449585, 1.1767370700836182, -1.7069061994552612, 0.24563923478126526, 1.1699823141098022, 1.7055377960205078, -0.8476811647415161, -0.0926055982708931, 0.7805234789848328, 0.8345732688903809, -0.8705499172210693, 0.3555912971496582, -1.6087746620178223,
  };
  static neuron_t neuron2 = {weights2, -0.10081951320171356};
  neuronas[2]=neuron2;
    
  static const float weights3[]={
    0.9586364030838013, -0.5918670892715454, -0.8268718719482422, -0.5351220369338989, -0.1265677511692047, -0.7040252089500427, 1.7245376110076904, 0.9141115546226501, -1.474987268447876, -0.768420398235321, -0.3163953423500061, 0.5411729216575623, 1.4656076431274414, 1.385102391242981, 0.644230842590332, 0.9081235527992249,
  };
  static neuron_t neuron3 = {weights3, -0.019207650795578957};
  neuronas[3]=neuron3;
    
  static const float weights4[]={
    -0.5615041255950928, 0.7283532619476318, 0.4729697108268738, 0.920714795589447, -0.9408352971076965, 0.5813233256340027, -1.0545464754104614, -0.05480881780385971, -0.6937531232833862, 1.1068625450134277, 0.9769448637962341, -0.8039473295211792, -0.06745266169309616, 1.6749941110610962, -0.981185793876648, 0.4785040318965912,
  };
  static neuron_t neuron4 = {weights4, -0.25782787799835205};
  neuronas[4]=neuron4;
    
  static const float weights5[]={
    -1.3140687942504883, 1.548262357711792, -0.8949456810951233, 0.903134286403656, -0.7919571995735168, 0.30244114995002747, 1.1712367534637451, -0.9127393364906311, -1.2956255674362183, -0.4589901864528656, -0.819442093372345, 1.027680516242981, -0.10690371692180634, -0.7447956800460815, 0.321438729763031, 0.4247226119041443,
  };
  static neuron_t neuron5 = {weights5, 0.048550352454185486};
  neuronas[5]=neuron5;
    
  static const float weights6[]={
    -0.06192854046821594, 1.1719645261764526, -1.4986677169799805, -1.3886016607284546, -0.43126314878463745, 0.697017252445221, 0.8828915357589722, -0.02893613837659359, 0.15545131266117096, 0.6970376372337341, 1.3320287466049194, -0.5969626307487488, -1.0229566097259521, -1.404591679573059, -1.0598986148834229, -0.20941556990146637,
  };
  static neuron_t neuron6 = {weights6, -0.1769881695508957};
  neuronas[6]=neuron6;
    
  static const float weights7[]={
    -1.5660865306854248, -1.0589048862457275, 1.4040898084640503, 1.3322529792785645, 0.5924639701843262, -0.07542720437049866, 1.2274309396743774, -0.17868342995643616, 0.643151581287384, 0.9497289657592773, 0.4090797007083893, -1.0805559158325195, 1.4365549087524414, -0.8477448225021362, 0.9176198840141296, -0.6033837199211121,
  };
  static neuron_t neuron7 = {weights7, 0.3632001280784607};
  neuronas[7]=neuron7;
    
  static const float weights8[]={
    1.2143865823745728, -0.620672881603241, 0.16463007032871246, -0.9651292562484741, 1.055778980255127, 1.0417497158050537, -0.49433550238609314, -0.46202901005744934, 1.5435491800308228, -0.30917298793792725, -1.1841320991516113, -1.1530100107192993, -0.5453038215637207, -0.9560359120368958, 0.25590282678604126, 0.9630212783813477,
  };
  static neuron_t neuron8 = {weights8, 0.1947120875120163};
  neuronas[8]=neuron8;
    
  static const float weights9[]={
    -0.10136929154396057, -1.2588450908660889, 0.8484535813331604, 0.3295719027519226, -0.029788313433527946, -0.9429451823234558, -1.55329167842865, -0.7127261161804199, -1.003599762916565, -1.0687618255615234, -0.8473684787750244, 1.0277771949768066, -1.289470911026001, 1.3185993432998657, 0.6875274181365967, 0.7048997282981873,
  };
  static neuron_t neuron9 = {weights9, -0.07753774523735046};
  neuronas[9]=neuron9;
    
  dense_layer_t layer= {10, neuronas};
  return layer;
}

void model_init(){
    conv_layer0 = init_conv_layer0(); // Capa convolucional 1
    conv_layer1 = init_conv_layer1(); // Capa convolucional 2
    dense_layer0 = init_dense_layer0(); //Capa densa 1
    dense_layer1 = init_dense_layer1(); //Capa densa 2
}



int model_predict(data_t input, flatten_data_t * results){
  data_t output;
  flatten_data_t f_input;
  
  // Capa 1: Conv 2D
  conv2d_layer(conv_layer0,input,&output);
    // Activation Layer 1: tanh
  tanh2d(output);
    input=output;
  
  // Capa 2: MaxPooling2D
  max_pooling_2d(2,2,input,&output);
  input=output;
  
  // Capa 3: Conv 2D
  conv2d_layer(conv_layer1,input,&output);
    // Activation Layer 3: tanh
  tanh2d(output);
    input=output;
  
  // Capa 4: Flatten
  flatten_data_t f_output;
  flatten_layer(output, &f_output);
  f_input=f_output;
  
  // Capa 5: Dense
  dense_forward(dense_layer0,f_input,&f_output);
  
  //Activación Capa 5: tanh
  tanh_flatten(f_output);
  f_input = f_output;
    
  // Capa 6: Dense
  dense_forward(dense_layer1,f_input,&f_output);
  
  //Activación Capa 6: softmax
  softmax(f_output);
    
  int result= argmax(f_output);
  *results = f_output;
  return result;
}
  