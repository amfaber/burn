use burn_core::data::dataloader::DataLoader;
use burn_core::tensor::backend::AutodiffBackend;
use burn_core::{
    lr_scheduler::LrScheduler, module::AutodiffModule, optim::GradientsAccumulator,
    tensor::backend::Backend,
};
use tracing_macro::scope;
use std::sync::Arc;

use crate::metric::processor::{Event, EventProcessor, LearnerItem};
use crate::{MultiDevicesTrainStep, TrainStep, ValidStep};
use crate::{components::LearnerComponents, learner::base::TrainingInterrupter};

/// A validation epoch.
#[derive(new)]
pub struct ValidEpoch<B: Backend, VI> {
    dataloader: Arc<dyn DataLoader<B, VI>>,
    epoch: usize,
    epoch_total: usize,
}

/// A training epoch.
#[derive(new)]
pub struct TrainEpoch<B: AutodiffBackend, TI> {
    dataloader: Vec<Arc<dyn DataLoader<B, TI>>>,
    epoch: usize,
    epoch_total: usize,
    grad_accumulation: Option<usize>,
}

impl<B: Backend, VI> ValidEpoch<B, VI> {
    /// Runs the validation epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to validate.
    /// * `processor` - The event processor to use.
    #[tracing::instrument(skip_all)]
    pub fn run<LC: LearnerComponents, VO>(
        &self,
        model: &LC::Model,
        processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
    ) where
        LC::EventProcessor: EventProcessor<ItemValid = VO>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<VI, VO>,
        LC::Backend: AutodiffBackend<InnerBackend = B>,
    {
        log::info!("Executing validation step for epoch {}", self.epoch);
        let model = model.valid();

        let mut iterator = self.dataloader.iter();
        let mut iteration = 0;

        while let Some(item) = iterator.next() {
            let progress = iterator.progress();
            iteration += 1;

            let item = model.step(item);
            let item = LearnerItem::new(
                item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                None,
            );

            processor.process_valid(Event::ProcessedItem(item));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        processor.process_valid(Event::EndEpoch(self.epoch));
    }
}

impl<B: AutodiffBackend, TI> TrainEpoch<B, TI> {
    /// Runs the training epoch.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    #[tracing::instrument(skip_all)]
    pub fn run<LC: LearnerComponents<Backend = B>, TO>(
        &mut self,
        mut model: LC::Model,
        mut optim: LC::Optimizer,
        scheduler: &mut LC::LrScheduler,
        processor: &mut LC::EventProcessor,
        interrupter: &TrainingInterrupter,
    ) -> (LC::Model, LC::Optimizer)
    where
        LC::EventProcessor: EventProcessor<ItemTrain = TO>,
        LC::Model: TrainStep<TI, TO>,
    {
        log::info!("Executing training step for epoch {}", self.epoch,);

        // Single device / dataloader
        let mut iterator = scope!("get dataloader", self.dataloader[0].iter());
        let mut iteration = 0;
        let mut accumulator = scope!("new acc", GradientsAccumulator::new());
        let mut accumulation_current = 0;

        while let Some(item) = scope!("iter.next", iterator.next()) {
            iteration += 1;
            let lr = scheduler.step();
            log::info!("Iteration {iteration}");

            let progress = scope!("progress", iterator.progress());
            let item = model.step(item);

            match self.grad_accumulation {
                Some(accumulation) => {
                    scope!("accumulate", accumulator.accumulate(&model, item.grads));
                    accumulation_current += 1;

                    if accumulation <= accumulation_current {
                        let grads = scope!("grads", accumulator.grads());
                        model = model.optimize(&mut optim, lr, grads);
                        accumulation_current = 0;
                    }
                }
                None => model = model.optimize(&mut optim, lr, item.grads),
            }

            let item = scope!("new learner item", LearnerItem::new(
                item.item,
                progress,
                self.epoch,
                self.epoch_total,
                iteration,
                Some(lr),
            ));

            scope!("process train 1", processor.process_train(Event::ProcessedItem(item)));

            if interrupter.should_stop() {
                log::info!("Training interrupted.");
                break;
            }
        }
        scope!("process train 2", processor.process_train(Event::EndEpoch(self.epoch)));

        self.epoch += 1;

        (model, optim)
    }
}

impl<B: AutodiffBackend, TI> TrainEpoch<B, TI> {
    /// Runs the training epoch on multiple devices.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to train.
    /// * `optim` - The optimizer to use.
    /// * `lr_scheduler` - The learning rate scheduler to use.
    /// * `processor` - The event processor to use.
    /// * `devices` - The devices to use.
    ///
    /// # Returns
    ///
    /// The trained model and the optimizer.
    pub fn run_multi_device<LC: LearnerComponents<Backend = B>, TO>(
        &mut self,
        mut model: LC::Model,
        mut optim: LC::Optimizer,
        lr_scheduler: &mut LC::LrScheduler,
        processor: &mut LC::EventProcessor,
        devices: Vec<<LC::Backend as Backend>::Device>,
        interrupter: &TrainingInterrupter,
    ) -> (LC::Model, LC::Optimizer)
    where
        LC::EventProcessor: EventProcessor<ItemTrain = TO>,
        LC::Model: TrainStep<TI, TO>,
        TO: Send + 'static,
        TI: Send + 'static,
    {
        log::info!(
            "Executing training step for epoch {} on devices {:?}",
            self.epoch,
            devices
        );

        let mut iterators = self.dataloader.iter().map(|d| d.iter()).collect::<Vec<_>>();
        let mut iteration = 0;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0;

        let accumulation = self.grad_accumulation.unwrap_or(1) * devices.len();
        let step = MultiDevicesTrainStep::new(&devices);

        // The main device is always the first in the list.
        let device_main = devices.first().expect("A minimum of one device.").clone();
        let mut interrupted = false;

        loop {
            let (items, progress) = step.step(iterators.as_mut_slice(), &model);
            if items.is_empty() {
                break;
            }

            for item in items {
                iteration += 1;
                let lr = lr_scheduler.step();

                // TODO: aggregate multi device (all-reduce)
                let grads = item.grads.to_device(&device_main, &model);

                accumulator.accumulate(&model, grads);
                accumulation_current += 1;

                if accumulation <= accumulation_current {
                    let grads = accumulator.grads();
                    model = model.optimize(&mut optim, lr, grads);
                    accumulation_current = 0;
                }

                let item = LearnerItem::new(
                    item.item,
                    progress.clone(),
                    self.epoch,
                    self.epoch_total,
                    iteration,
                    Some(lr),
                );

                processor.process_train(Event::ProcessedItem(item));

                if interrupter.should_stop() {
                    log::info!("Training interrupted.");
                    interrupted = true;
                    break;
                }
            }

            if interrupted {
                break;
            }
        }

        processor.process_train(Event::EndEpoch(self.epoch));

        self.epoch += 1;

        (model, optim)
    }
}
