"""Control-flow tests only: never loads Sana or performs optimizer updates."""
import json
from pathlib import Path
import tempfile
import unittest
import yaml
from train_pilot import ROOT, lock_directory, plan, sample_for_step

class PilotTests(unittest.TestCase):
    def test_epoch_shuffle_and_resume_position(self):
        rows=[{'id':str(i)} for i in range(17)]
        sequence=[sample_for_step(rows,i,42)['id'] for i in range(40)]
        self.assertEqual(len(set(sequence[:17])),17)
        self.assertEqual(len(set(sequence[17:34])),17)
        self.assertNotEqual(sequence[:17],sequence[17:34])
        self.assertEqual(sequence[23:],[sample_for_step(rows,i,42)['id'] for i in range(23,40)])
    def test_lock_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            with lock_directory(output):
                with self.assertRaises(FileExistsError):
                    with lock_directory(output): pass
                self.assertTrue((output/'.running').exists())
            self.assertFalse((output/'.running').exists())
    def test_fixed_heldout_validation_selection(self):
        from validate_pilot import select_validation
        config=Path(__file__).with_name('pilot_config.yaml')
        cfg,rows,_,_=plan(config)
        chosen=select_validation(cfg)
        self.assertEqual(chosen,select_validation(cfg))
        self.assertEqual([r['category'] for r in chosen],cfg['validation']['categories'])
        self.assertTrue(all(r['split']=='validation' for r in chosen))
        self.assertFalse({r['split_group'] for r in chosen} & {r['split_group'] for r in rows})

    def test_validated_plan_and_step_cap(self):
        config=Path(__file__).with_name('pilot_config.yaml')
        cfg,rows,identity,fingerprint=plan(config)
        self.assertEqual(len(rows),192)
        self.assertEqual(cfg['checkpoint_steps'],[50,100,200])
        self.assertEqual(len(fingerprint),64)
        with tempfile.TemporaryDirectory() as d:
            cfg['max_steps']=201
            path=Path(d)/'invalid.yaml'; path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError): plan(path)

if __name__=='__main__': unittest.main()
