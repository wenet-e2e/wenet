#pragma once

#include <c10/core/thread_pool.h>
#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

#include <atomic>
#include <thread>

namespace torch {
namespace distributed {
namespace rpc {

constexpr auto kDefaultNumSendRecvThreads = 4;

struct ProcessGroupRpcBackendOptions : public RpcBackendOptions {
  ProcessGroupRpcBackendOptions(
      int num_send_recv_threads,
      float rpc_timeout,
      std::string init_method)
      : RpcBackendOptions(rpc_timeout, init_method),
        numSendRecvThreads(num_send_recv_threads) {
    TORCH_CHECK(
        num_send_recv_threads > 0,
        "Cannot create ProcessGroup RPC backend with ",
        num_send_recv_threads,
        " threads in the thread-pool.");
  }

  int numSendRecvThreads;
};

// SendWork and RecvWork will be put into a task queue, and later picked up by
// worker threads from the same ThreadPool.
struct SendWork {
  SendWork(const WorkerInfo& to, Message&& message)
      : to_(to), message_(message) {}

  const WorkerInfo& to_;
  Message message_;
};

// SendWork wraps a Message and RecvWork wraps a Tensor. The difference here is
// to allow us to run serialization/deserialization in the worker threads.
struct RecvWork {
  RecvWork(
      const WorkerInfo& from,
      MessageType type,
      int64_t id,
      torch::Tensor&& payload)
      : from_(from), type_(type), id_(id), payload_(payload) {}

  const WorkerInfo& from_;
  const MessageType type_;
  const int64_t id_;
  torch::Tensor payload_;
};

class TORCH_API ProcessGroupAgent : public RpcAgent {
 public:
  ProcessGroupAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string workerName,
      c10::intrusive_ptr<::c10d::ProcessGroup> pg,
      int numSendRecvThreads,
      std::chrono::milliseconds rpcTimeout,
      std::unique_ptr<RequestCallback> cb);

  const WorkerInfo& getWorkerInfo(const std::string& workerName) const override;

  const WorkerInfo& getWorkerInfo(worker_id_t id) const override;

  std::vector<WorkerInfo> getWorkerInfos() const override;

  void join(bool shutdown = false) override;

  void sync() override;

  void startImpl() override;

  void shutdownImpl() override;

  ~ProcessGroupAgent() override;

  std::unordered_map<std::string, std::string> getMetrics() override;

 protected:
  // This method wraps the destination information and the message into a
  // SendWork object, and put the SendWork into a queue. Another thread will
  // consume SendWork from the queue and send it out.
  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      Message&& message,
      const float rpcTimeoutSeconds = kUnsetRpcTimeout,
      const std::unordered_map<c10::Device, c10::Device>& deviceMap = {})
      override;

  // put SendWork into a queue and notify the worker thread
  virtual void enqueueSend(SendWork work);
  // Bypass handleSend() logic and send a message to self rank
  virtual void sendToSelf(Message&& message);

 private:
  class MessageCounter {
   public:
    explicit MessageCounter(int worldSize);
    void increment(int dst);
    std::vector<int64_t> snapshot();

   private:
    std::vector<int64_t> counters_;
    std::mutex mutex_;
  };

  // TODO: this class should inherit from a MetricsTracker, and can be extended
  // to track num_sends, recvs, average size of messages, etc.
  struct AverageMetricsTracker {
    std::string key_;
    uint64_t currentSum_;
    uint64_t currentCount_;

    explicit AverageMetricsTracker(
        std::string key,
        uint64_t currentSum = 0,
        uint64_t currentCount = 0);

    void addData(uint64_t dataPoint);
    double computeAverage();
  };

  // The FutureInfo struct stores a shared_ptr to the future, as well as
  // additional information to manage timeouts and destination information,
  // which is needed for termination detection.
  struct FutureInfo {
    c10::intrusive_ptr<JitFuture> future_;
    steady_clock_time_point endTime_;
    int dstRank_;
    std::chrono::milliseconds timeout_;
    FutureInfo(
        c10::intrusive_ptr<JitFuture> future,
        const steady_clock_time_point& endTime,
        int dstRank,
        const std::chrono::milliseconds timeout)
        : future_(std::move(future)),
          endTime_(endTime),
          dstRank_(dstRank),
          timeout_(timeout) {}
    FutureInfo() = delete;
  };

  // handle a SendWork request. This serializes the payload inside the work
  // object, and sends the message to the receiver using the underlying
  // ProcessGroup.
  void handleSend(const SendWork& work);
  // put RecvWork into a queue and notify the worker thread
  void enqueueRecv(RecvWork work);
  // handle a RecvWork request. Return true if we should increment recvCounts,
  // false if not (i.e. if the RPC timed out and we are getting a result after
  // the timeout). This ensures that the messages accounted for in
  // hasPendingMessage() are tallied properly during a graceful shutdown.
  bool handleRecv(RecvWork& work);
  // Loop that receives and processes messages
  void listenLoopInternal();
  // Calls listenLoopInternal and handles errors such as timeouts on the
  // process group.
  void listenLoop();
  // exception_pointer correspnding to an exception raised in listenLoop (if
  // there is one), and lock to guard access.
  std::exception_ptr listenLoopException_;
  std::mutex listenLoopExceptionMutex_;
  // poll for timed out RPCs
  void pollTimedOutRPCs();
  // process timed out futures
  const std::vector<FutureInfo> processTimedOutFutures();
  // compute the remaining time for an RPC, given its end time.
  const std::chrono::milliseconds getRPCRemainingTime(
      const std::chrono::milliseconds& rpcEndTime) const;

  // a helper function to mark a future in the futures_ map with a message. The
  // future is marked with the passed in message, and then removed from the
  // futures_ map. It is also removed from the futureTimeouts_ map since these
  // maps are kept in sync.
  void markFutureWithError(Message& message);
  void markFutureWithError(int64_t id, std::string errorMsg);

  // Note [Termination Detection]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // RpcAgent implementations must properly detect termination. Otherwise, it
  // would result in message loss, RRef leak, or process hang. It is not
  // sufficient to just wait for the thread pool to finish processing all tasks
  // after all processes hit the join function. There could be nested rpc/remote
  // calls, meaning that an empty task queue in the thread pool does not mean
  // there will be no tasks added in the future. Moreover, in the listenLoop,
  // there is a period of time when the message has been received but not yet
  // inserted into the thread pool, which also suggests that the empty task
  // queue is not a good indicator for termination.
  //
  // To detect termination, each ProcessGroupAgent maintains a sent message
  // counter and a received message counter. The sent message counter is
  // incremented whenever a message is sent, and the receive message counter is
  // only incremented when a message has been processed. During termination, all
  // ProcessGroupAgent instances run an allgather to collect counters from all
  // peers, which means that all agents will have a consistent view on the
  // message count snapshot. They would only terminate if all sent/received
  // message counters match.
  bool hasPendingMessage();

  int64_t nextId() {
    return ++nextId_;
  }

  c10::intrusive_ptr<::c10d::ProcessGroup> pg_;
  // worker name -> rank
  std::unordered_map<std::string, worker_id_t> nameMap_;
  std::vector<WorkerInfo> allWorkerInfo_;
  // record the number of messages sent to and received from each peer. The recv
  // counter is only marked after the message is processed. Join uses allgather
  // to collect all counts from all peers, uses these counters to detect global
  // termination and only exit when all sent messages are processed.
  MessageCounter sendCounts_;
  MessageCounter recvCounts_;

  std::atomic<int64_t> nextId_;
  // one mutex per ProcessGroup rank, as ProcessGroup::send is not thread-safe
  // when using the same tag.
  std::vector<std::mutex> sendMutexes_;
  std::thread listenerThread_;
  // A thread to poll existing futures and check for timed out ones.
  std::thread futureTimeoutThread_;
  // Lock and shared ptr to currently pending work, set in listenloop() and
  // interruptible in shutdown().
  std::mutex recvWorkMutex_;
  c10::intrusive_ptr<c10d::ProcessGroup::Work> recvWork_;
  // Map of dst rank to current oustanding sends that we are waiting on. In the
  // case of a call to ::shutdown() while we are still waiting on these sends,
  // the pending sends contained in this map will be aborted, allowing the
  // waiting thread to be unblocked.
  std::unordered_map<
      worker_id_t,
      std::set<c10::intrusive_ptr<c10d::ProcessGroup::Work>>>
      currentPendingSends_;
  // Lock to serialize access to the above map.
  std::mutex pendingSendMutex_;
  // A threadPool that processing both SendWork and RecvWork. There are two
  // motivations for adding a ThreadPool:
  // (1) RPC serialization/deserialization and processing can be expensive,
  //     hence using multiple threads to speed it up.
  // (2) The current RPC API does not support asynchronous UDFs, e.g., UDFs can
  //     not yield in the middle of execution to wait for IO, and resume the IO
  //     is done. This would result in deadlocks when we have nested RPC calls.
  //     NB: Ideally, this should be addressed by supporting asynchronous UDF.
  //         This is just a temporary solution for (2).
  ThreadPool threadPool_;
  // Atomic to indicate whether the timeout thread is enabled.
  std::atomic<bool> timeoutThreadEnabled_;
  // Mapping of request id to FutureInfo struct.
  std::unordered_map<int64_t, FutureInfo> futures_;
  // A map to keep track of when futures time out. The map is keyed by the time
  // (millisecond level precision) the future will expire. This is so that timed
  // out futures can be efficiently cleaned up, and we can quickly exit if we
  // find a future that has not timed out. The values correspond to an
  // unordered_set of future ids that started at that time. This map must be
  // kept in sync with the above futures_ map.
  std::map<steady_clock_time_point, std::unordered_set<int64_t>>
      futureTimeouts_;
  mutable std::mutex futureMutex_;
  mutable std::condition_variable futureCV_;
  // CV to wake up watchdog thread that watches for timed out futures.
  std::condition_variable futureTimeoutCV_;
  // Metrics tracked for ProcessGroupAgent.
  enum ProcessGroupAgentMetrics {
    GIL_WAIT_TIME = 0,

    N_METRICS,
  };
  std::mutex metricsMutex_;
  std::vector<std::unique_ptr<AverageMetricsTracker>> metrics_;
  void addGilWaitTime(const std::chrono::microseconds gilWaitTime) override;

  std::atomic<int32_t> clientActiveCalls_{0};
  std::atomic<int32_t> serverActiveCalls_{0};
  std::atomic<int32_t> serverActiveAsyncCalls_{0};
};

} // namespace rpc
} // namespace distributed
} // namespace torch
