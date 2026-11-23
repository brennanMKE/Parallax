package ai

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// ErrNoSlot is a pool with no free request slot on any healthy endpoint
// within its wait.
var ErrNoSlot = errors.New("no model free")

// Endpoint is one System 1 server in a Pool.
type Endpoint struct {
	// Name identifies the endpoint in logs and stats.
	Name  string
	Model Asker
	// Health, if set, tracks whether the endpoint answers. The pool skips
	// it while it's down.
	Health Health
	// OnError, if set, hears about this endpoint's failed requests (not
	// busy answers), so its health can mark it down at once.
	OnError func(error)
	// OnAnswer, if set, hears about each request it answered.
	OnAnswer func()
	// Slots is how many requests it may have in flight. 0 means 1.
	Slots int

	slots    chan struct{}
	answered atomic.Int64
	failed   atomic.Int64
}

// Pool spreads System 1 requests over several endpoints, in order of
// preference: a request goes to the first healthy endpoint with a free
// slot, so the fastest server takes the load until it's full and the next
// takes the overflow. A request that fails (or gets busy) on one endpoint
// is tried once more on another, within the same deadline.
type Pool struct {
	endpoints []*Endpoint
	// Wait bounds how long a request waits for a free slot anywhere.
	Wait time.Duration

	mu    sync.Mutex
	freed chan struct{} // closed and replaced whenever a slot frees
}

// NewPool makes a pool of endpoints, most preferred first.
func NewPool(wait time.Duration, endpoints ...*Endpoint) *Pool {
	for _, e := range endpoints {
		e.slots = make(chan struct{}, max(e.Slots, 1))
	}
	return &Pool{endpoints: endpoints, Wait: wait, freed: make(chan struct{})}
}

// Endpoints lists the pool's endpoints in order.
func (p *Pool) Endpoints() []*Endpoint { return p.endpoints }

func (e *Endpoint) up() bool { return e.Health == nil || e.Health.Up() }

// Up implements Health: the pool is up while any endpoint is.
func (p *Pool) Up() bool {
	for _, e := range p.endpoints {
		if e.up() {
			return true
		}
	}
	return false
}

// Check implements Health: it checks every endpoint and succeeds if any
// answers.
func (p *Pool) Check(ctx context.Context) error {
	errs := make([]error, len(p.endpoints))
	var wg sync.WaitGroup
	for i, e := range p.endpoints {
		if e.Health == nil {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = e.Health.Check(ctx)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err == nil {
			return nil
		}
	}
	return errors.Join(errs...)
}

// InFlight is how many requests are in flight across the pool.
func (p *Pool) InFlight() int {
	n := 0
	for _, e := range p.endpoints {
		n += len(e.slots)
	}
	return n
}

// EndpointStats is one endpoint's use so far.
type EndpointStats struct {
	Name             string
	Up               bool
	InFlight         int
	Answered, Failed int64
}

// Stats reads every endpoint's counters.
func (p *Pool) Stats() []EndpointStats {
	out := make([]EndpointStats, len(p.endpoints))
	for i, e := range p.endpoints {
		out[i] = EndpointStats{e.Name, e.up(), len(e.slots), e.answered.Load(), e.failed.Load()}
	}
	return out
}

// Ask implements Asker.
func (p *Pool) Ask(ctx context.Context, prompt Prompt) (Answers, error) {
	tried := map[*Endpoint]bool{}
	wait := p.Wait
	var lastErr error
	for range 2 {
		e, release := p.acquire(ctx, tried, wait)
		if e == nil {
			break
		}
		tried[e] = true
		ans, err := e.Model.Ask(ctx, prompt)
		release()
		if err == nil {
			e.answered.Add(1)
			if e.OnAnswer != nil {
				e.OnAnswer()
			}
			return ans, nil
		}
		lastErr = err
		if ctx.Err() != nil {
			// Out of time: no point trying another endpoint.
			return nil, err
		}
		if !errors.Is(err, ErrBusy) {
			e.failed.Add(1)
			if e.OnError != nil {
				e.OnError(err)
			}
		}
		// A retry takes only a slot that's free now.
		wait = 0
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, ErrNoSlot
}

// acquire takes a slot on the first healthy, untried endpoint that has one,
// waiting up to wait for one to free.
func (p *Pool) acquire(ctx context.Context, skip map[*Endpoint]bool, wait time.Duration) (*Endpoint, func()) {
	var timeout <-chan time.Time
	if wait > 0 {
		t := time.NewTimer(wait)
		defer t.Stop()
		timeout = t.C
	}
	for {
		p.mu.Lock()
		freed := p.freed
		p.mu.Unlock()
		for _, e := range p.endpoints {
			if skip[e] || !e.up() {
				continue
			}
			select {
			case e.slots <- struct{}{}:
				return e, func() { <-e.slots; p.signal() }
			default:
			}
		}
		if wait <= 0 {
			return nil, nil
		}
		select {
		case <-freed:
		case <-timeout:
			return nil, nil
		case <-ctx.Done():
			return nil, nil
		}
	}
}

func (p *Pool) signal() {
	p.mu.Lock()
	close(p.freed)
	p.freed = make(chan struct{})
	p.mu.Unlock()
}
