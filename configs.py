import datetime
import os

class ModelConfigFactory():
    @staticmethod
    def create_model_config(args):
        if args.dataset == 'assist2009':
            return Assist2009Config(args).get_args()
        elif args.dataset == 'assist2015':
            return Assist2015Config(args).get_args()
        elif args.dataset == 'statics2011':
            return StaticsConfig(args).get_args()
        elif args.dataset == 'synthetic':
            return SyntheticConfig(args).get_args()
        elif args.dataset == 'fsai':
            return FSAIConfig(args).get_args()
        else:
            raise ValueError("The '{}' is not available".format(args.dataset))


class ModelConfig():
    def __init__(self, args):
        self.default_setting = self.get_default_setting()
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")

        self.args = args
        self.args_dict = vars(self.args)
        for arg in self.args_dict.keys():
            self._set_attribute_value(arg, self.args_dict[arg])  

        self.set_result_log_dir()
        self.set_checkpoint_dir()
        self.set_tensorboard_dir() 
    
    def get_args(self):
        return self.args

    def get_default_setting(self):
        default_setting = {}
        return default_setting

    def _set_attribute_value(self, arg, arg_value):
        self.args_dict[arg] = arg_value \
            if arg_value is not None \
            else self.default_setting.get(arg)
    
    def _get_model_config_str(self):
        model_config = 'b' + str(self.args.batch_size) \
                    + '_m' + str(self.args.memory_size) \
                    + '_q' + str(self.args.key_memory_state_dim) \
                    + '_qa' + str(self.args.value_memory_state_dim) \
                    + '_f' + str(self.args.summary_vector_output_dim)
        return model_config

    def set_result_log_dir(self):
        result_log_dir = os.path.join(
            './results',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('result_log_dir', result_log_dir)

    def set_checkpoint_dir(self):
        checkpoint_dir = os.path.join(
            './models',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('checkpoint_dir', checkpoint_dir)
    
    def set_tensorboard_dir(self):
        tensorboard_dir = os.path.join(
            './tensorboard',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('tensorboard_dir', tensorboard_dir)
        

class Assist2009Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 110,
            'data_dir': './data/assist2009_updated',
            'data_name': 'assist2009_updated',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2015Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 100,
            'data_dir': './data/assist2015',
            'data_name': 'assist2015',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class StaticsConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 1223,
            'data_dir': './data/STATICS',
            'data_name': 'STATICS',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class SyntheticConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 50,
            'data_dir': './data/synthetic',
            'data_name': 'synthetic',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class FSAIConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 2266,
            'data_dir': './data/fsaif1tof3',
            'data_name': 'fsaif1tof3',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting